"""From live events to vehicle profiles in the database (live mode, FR5/FR6).

`PassRecorder` sits between the event assembler (src/live_events.py) and
the database (src/db.py truck_passes):

- Every tracked detection's vehicle-type probabilities and every saved
  OCR crop are handed to it as frames are processed (`add_types`,
  `add_crop`).
- On each assembler update, *provisional* events are written straight
  away with what is known without OCR (cameras, times, vehicle type so
  far), so a dashboard can show a truck while it is still in view. An
  event that was provisional and has disappeared (the assembler split or
  joined passes differently once more tracks ended) is deleted.
- *Final* events are submitted for OCR (`submit(job)`; in
  scripts/run_live.py a separate process with PaddleOCR). OCR is the slow
  step, so it runs off the main loop.
- When an OCR result comes back (`on_reads`), codes, plates and vehicle
  type are fused into a profile (src.profiles) and written as final.
  Container codes are then checked against the trucks finalised within
  the last few minutes (src.fusion.resolve_shared_codes); a neighbour
  whose codes change is rewritten too.

Vehicle type is fused with cameras weighted equally (`per_camera=True`).
That variant was chosen on the 12 Aug hour, which is the development set
since the live streams arrived (see the brief), so using it on new days
is fine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from .db import delete_pass, upsert_pass
from .events import Event
from .fusion import FusedRead, resolve_shared_codes
from .live_events import LiveUpdate, event_id
from .live_ocr import EventReads, SpooledCrop, fuse_event
from .profiles import VehicleProfile, build_profile, refresh_codes
from .vehicle_type import FrameType, fuse_vehicle_type


@dataclass
class OcrJob:
    """One final event to read."""
    pass_uid: str
    crops: list[SpooledCrop]


@dataclass
class _Final:
    event: Event
    profile: VehicleProfile
    code: FusedRead = field(default_factory=lambda: FusedRead(None, "failed"))


class PassRecorder:
    def __init__(self, engine, submit: Callable[[OcrJob], None], *, source: str = "live",
                 share_window_s: float = 180.0, keep_s: float = 1800.0,
                 clock: Callable[[], float] = None):
        import time
        self.engine = engine
        self.submit = submit
        self.source = source
        self.share_window_s = share_window_s
        self.keep_s = keep_s
        self.clock = clock or time.time
        self.types: dict[tuple[str, int], dict[int, FrameType]] = defaultdict(dict)
        self.crops: dict[tuple[str, int], list[SpooledCrop]] = defaultdict(list)
        self.provisional: dict[str, VehicleProfile] = {}
        self.provisional_events: dict[str, Event] = {}
        self.pending: dict[str, Event] = {}          # submitted for OCR, result not back yet
        self.finals: dict[str, _Final] = {}
        self.written: dict[str, tuple] = {}          # pass_uid -> last written state

    # -- inputs from the tracking stage ------------------------------------------
    def add_types(self, camera: str, types: dict[tuple[int, int], FrameType]) -> None:
        for (tid, frame), ft in types.items():
            self.types[(camera, tid)][frame] = ft

    def add_crop(self, crop: SpooledCrop) -> None:
        self.crops[(crop.camera, crop.track_id)].append(crop)

    # -- event updates ------------------------------------------------------------
    def _vehicle_type(self, e: Event, detections) -> object:
        frames = []
        for p in e.passes:
            by_frame = self.types.get((p.camera, p.track_id), {})
            kept = {d.frame for d in detections.get((p.camera, p.track_id), [])}
            frames += [ft for f, ft in by_frame.items() if f in kept]
        return fuse_vehicle_type(frames, per_camera=True)

    def _crops(self, e: Event, detections) -> list[SpooledCrop]:
        out = []
        for p in e.passes:
            kept = {d.frame for d in detections.get((p.camera, p.track_id), [])}
            out += [c for c in self.crops.get((p.camera, p.track_id), []) if c.frame in kept]
        return out

    def _write(self, profile: VehicleProfile, status: str) -> None:
        state = (status, profile.codes, profile.vehicle_type, round(profile.vehicle_type_conf, 2),
                 profile.cameras, profile.plate, profile.trailer_plate, profile.end_time,
                 tuple(profile.notes))
        if self.written.get(profile.pass_uid) == state:
            return
        upsert_pass(self.engine, profile, status=status, source=self.source, updated_at=self.clock())
        self.written[profile.pass_uid] = state

    def on_update(self, up: LiveUpdate) -> None:
        """Write provisional events, submit final ones for OCR."""
        seen = set()
        for e in up.provisional:
            uid = event_id(e)
            seen.add(uid)
            p = build_profile(uid, e.entry_time, e.end, tuple(e.cameras),
                              vtype=self._vehicle_type(e, up.detections))
            self.provisional[uid] = p
            self.provisional_events[uid] = e
            self._write(p, "provisional")
        for e in up.finalized:
            uid = event_id(e)
            seen.add(uid)
            self.provisional.pop(uid, None)
            self.provisional_events.pop(uid, None)
            # Until OCR is back the pass stays provisional, with its final passes and type.
            p = build_profile(uid, e.entry_time, e.end, tuple(e.cameras),
                              vtype=self._vehicle_type(e, up.detections))
            self.finals[uid] = _Final(e, p)
            self._write(p, "provisional")
            self.pending[uid] = e
            self.submit(OcrJob(uid, self._crops(e, up.detections)))
        for uid in [u for u in self.provisional if u not in seen]:
            del self.provisional[uid]                 # re-assembled differently
            self.provisional_events.pop(uid, None)
            delete_pass(self.engine, uid)
            self.written.pop(uid, None)

    def on_reads(self, pass_uid: str, reads: EventReads) -> VehicleProfile:
        """OCR result of a final event: fuse, resolve shared codes, write final."""
        f = self.finals[pass_uid]
        self.pending.pop(pass_uid, None)
        code, plate, trailer = fuse_event(reads)
        e = f.event
        vt_profile = f.profile
        f.code = code
        f.profile = build_profile(pass_uid, e.entry_time, e.end, tuple(e.cameras), code=code,
                                  plate=plate, trailer=trailer,
                                  vtype=_FusedTypeView(vt_profile))
        # one container, one truck: among recently finalised trucks
        window = timedelta(seconds=self.share_window_s)
        near = {u: x for u, x in self.finals.items()
                if u not in self.pending and abs(x.event.entry_time - e.entry_time) <= 2 * window}
        resolve_shared_codes({u: x.code for u, x in near.items()},
                             {u: x.event.entry_time for u, x in near.items()},
                             window_s=self.share_window_s)
        for u, x in near.items():
            changed = refresh_codes(x.profile, x.code)
            if u == pass_uid or changed:
                self._write(x.profile, "final")
        self._prune()
        return f.profile

    def _prune(self) -> None:
        """Forget finals, types and crop lists older than `keep_s` (by event time)."""
        if not self.finals:
            return
        newest = max(x.event.end for x in self.finals.values())
        cutoff = newest - timedelta(seconds=self.keep_s)
        for u in [u for u, x in self.finals.items() if x.event.end < cutoff and u not in self.pending]:
            e = self.finals.pop(u).event
            self.written.pop(u, None)
            for p in e.passes:
                self.types.pop((p.camera, p.track_id), None)
                self.crops.pop((p.camera, p.track_id), None)

    def drop_crops_before(self, t_by_camera: dict[str, float], keep_s: float) -> list[SpooledCrop]:
        """Crop lists of tracks whose newest crop is older than `keep_s` (video seconds) on
        their camera and that no pending or provisional event uses; returns them for deletion."""
        used = set()
        for e in list(self.pending.values()) + list(self.provisional_events.values()):
            used |= {(p.camera, p.track_id) for p in e.passes}
        out = []
        for key in list(self.crops):
            cam, _ = key
            cs = self.crops[key]
            if key in used or not cs or cam not in t_by_camera:
                continue
            if t_by_camera[cam] - max(c.t for c in cs) > keep_s:
                out += self.crops.pop(key)
                self.types.pop(key, None)
        return out


class _FusedTypeView:
    """The vehicle type already fused for a profile, in FusedType's shape."""

    def __init__(self, p: VehicleProfile):
        self.vehicle_type = p.vehicle_type
        self.conf = p.vehicle_type_conf
