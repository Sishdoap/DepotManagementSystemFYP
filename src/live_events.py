"""Assemble checkpoint events while the cameras are still streaming (live mode).

The batch pipeline (src/events.py) sees a whole recording: every track is
complete before passes are filtered and assembled. Live, a truck is still
in view while its first cameras are done with it, so this module re-runs
the *same* `extract_events` on the tracks that have ended so far, every
time it is asked (`update`). There is deliberately no second assembly
algorithm: replaying a recording through it gives exactly the batch events
(tests/test_live_events.py replays the 12 Aug tracks to check that).

- A **track has ended** when its camera has not seen it for
  `track_timeout_s` (ByteTrack keeps a lost track for a few seconds; the
  footage is sampled at 4 fps).
- An event is **provisional** while something could still join it and
  **final** once nothing can: by the assembly rule a pass joins an event
  only within `max_lag_s` of its entry time or within the camera's merge
  gap of its previous fragment, so an event is final when the slowest
  camera's clock is past both (plus the track timeout and the clock
  tolerance). On the footage that is ~45-60 s after the truck leaves the
  last camera. Final events are reported once and never change.
- "Now" is the *slowest* camera's wall-clock time, so a camera that lags
  (network) cannot have its passes cut off.
- Memory stays bounded: tracks of final events older than `keep_s` are
  dropped. The re-run then no longer sees them; that cannot change any
  later event, because nothing can join an event that old.

Event ids are stable across updates: `<entry time>_<camera>_<track>` of the
event's opening pass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Optional

from .events import Event, extract_events
from .footage import CameraConfig, Detection, RecordingInfo, wall_clock


def event_id(e: Event) -> str:
    p = e.passes[0]
    return f"{e.entry_time:%Y%m%d-%H%M%S}_{p.camera}_{p.track_id}"


@dataclass
class LiveUpdate:
    provisional: list[Event] = field(default_factory=list)   # may still change
    finalized: list[Event] = field(default_factory=list)     # final since the last update
    now: Optional[datetime] = None


class LiveEventAssembler:
    """Feed tracked detections per camera; `update()` returns events so far."""

    def __init__(self, configs: Mapping[str, CameraConfig], recordings: Mapping[str, RecordingInfo],
                 *, track_timeout_s: float = 5.0, max_lag_s: float = 45.0, merge_gap_s: float = 25.0,
                 clock_tolerance_s: float = 2.0, keep_s: float = 900.0, **assembly_kwargs):
        self.configs = configs
        self.recordings = recordings
        self.track_timeout_s = track_timeout_s
        self.keep_s = keep_s
        self.assembly_kwargs = dict(max_lag_s=max_lag_s, merge_gap_s=merge_gap_s,
                                    clock_tolerance_s=clock_tolerance_s, **assembly_kwargs)
        gaps = [merge_gap_s] + [c.merge_gap_s for c in configs.values() if c.merge_gap_s is not None]
        # Seconds after an event's last pass / entry time when nothing more can join it.
        self.final_after_end_s = max(gaps) + track_timeout_s + clock_tolerance_s
        self.final_after_entry_s = max_lag_s + track_timeout_s + clock_tolerance_s
        self.tracks: dict[str, dict[int, list[Detection]]] = defaultdict(lambda: defaultdict(list))
        self.latest_t: dict[str, float] = {}           # camera -> video seconds of its latest frame
        self.final_ids: set[str] = set()
        self.final_events: list[Event] = []

    # -- input -----------------------------------------------------------------
    def add_frame(self, camera: str, t: float, detections: list[Detection]):
        """One processed frame of `camera` at video time `t` (detections may be empty)."""
        if camera not in self.configs or camera not in self.recordings:
            return
        self.latest_t[camera] = max(t, self.latest_t.get(camera, t))
        for d in detections:
            if d.track_id is not None:
                self.tracks[camera][d.track_id].append(d)

    # -- clocks ----------------------------------------------------------------
    def wall(self, camera: str, t: float) -> datetime:
        return wall_clock(self.recordings[camera], t, self.configs[camera].offset_at(t))

    def now(self) -> Optional[datetime]:
        """Wall-clock time every camera has reached (the slowest camera)."""
        cams = [c for c in self.configs if self.configs[c].role != "ignore" and c in self.recordings]
        if not cams or any(c not in self.latest_t for c in cams):
            return None
        return min(self.wall(c, self.latest_t[c]) for c in cams)

    # -- assembly --------------------------------------------------------------
    def ended_tracks(self) -> dict[str, list[Detection]]:
        out = {}
        for cam, by_id in self.tracks.items():
            latest = self.latest_t.get(cam, float("-inf"))
            out[cam] = [d for dets in by_id.values()
                        if latest - dets[-1].t >= self.track_timeout_s for d in dets]
        return out

    def active_track_starts(self) -> list[datetime]:
        """Wall-clock start of every track still in progress."""
        out = []
        for cam, by_id in self.tracks.items():
            latest = self.latest_t.get(cam, float("-inf"))
            out += [self.wall(cam, dets[0].t) for dets in by_id.values()
                    if latest - dets[-1].t < self.track_timeout_s]
        return out

    def update(self, final_all: bool = False) -> LiveUpdate:
        """Re-assemble; `final_all` finalises everything (end of a replay)."""
        now = self.now()
        tracks = self.tracks_snapshot() if final_all else self.ended_tracks()
        events = extract_events(tracks, self.configs, self.recordings, **self.assembly_kwargs).events
        result = LiveUpdate(now=now)
        active_starts = [] if final_all else self.active_track_starts()
        tol = timedelta(seconds=self.assembly_kwargs["clock_tolerance_s"])
        for e in events:
            eid = event_id(e)
            if eid in self.final_ids:
                continue
            # A track still in progress that began while this truck was in view
            # may belong to it (e.g. a truck standing at the barrier). Tracks
            # that began earlier (a parked chassis tracked for an hour) don't.
            pending = any(e.first_seen - tol <= s <= e.end + tol for s in active_starts)
            done = final_all or (now is not None and not pending
                                 and (now - e.end).total_seconds() > self.final_after_end_s
                                 and (now - e.entry_time).total_seconds() > self.final_after_entry_s)
            if done:
                self.final_ids.add(eid)
                self.final_events.append(e)
                result.finalized.append(e)
            else:
                result.provisional.append(e)
        if now is not None and not final_all:
            self._forget_before(now - timedelta(seconds=self.keep_s))
        return result

    def replay(self, tracks: Mapping[str, list[Detection]], *, frame_step_s: float = 0.25,
               update_every_s: float = 5.0) -> list[tuple[datetime, Event]]:
        """Feed recorded tracks frame by frame in wall-clock order, as a live
        stream would deliver them; returns (time finalised, event) pairs, with
        everything left finalised at the end. Used to test live mode against
        the batch pipeline on the same tracks."""
        import heapq

        by_frame = {cam: defaultdict(list) for cam in tracks}
        for cam, dets in tracks.items():
            for d in dets:
                by_frame[cam][round(d.t / frame_step_s)].append(d)
        heap = []
        for cam in tracks:
            if cam not in self.recordings or cam not in self.configs or not by_frame[cam]:
                continue
            last = max(by_frame[cam]) + int(4 * self.final_after_end_s / frame_step_s)
            heapq.heappush(heap, (self.wall(cam, 0.0), cam, 0, last))
        out, next_update = [], None
        while heap:
            w, cam, k, last = heapq.heappop(heap)
            self.add_frame(cam, k * frame_step_s, by_frame[cam].get(k, []))
            if k < last:
                heapq.heappush(heap, (self.wall(cam, (k + 1) * frame_step_s), cam, k + 1, last))
            if next_update is None or w >= next_update:
                next_update = w + timedelta(seconds=update_every_s)
                out += [(w, e) for e in self.update().finalized]
        out += [(None, e) for e in self.update(final_all=True).finalized]
        return out

    def tracks_snapshot(self) -> dict[str, list[Detection]]:
        return {cam: [d for dets in by_id.values() for d in dets] for cam, by_id in self.tracks.items()}

    def _forget_before(self, cutoff: datetime):
        """Drop tracks that ended before `cutoff` and belong to final events (or none)."""
        live_keys = set()
        for e in self.final_events:
            if e.end >= cutoff:
                live_keys |= {(p.camera, p.track_id) for p in e.passes}
        for cam, by_id in self.tracks.items():
            for tid in [tid for tid, dets in by_id.items()
                        if self.wall(cam, dets[-1].t) < cutoff and (cam, tid) not in live_keys]:
                del by_id[tid]
        self.final_events = [e for e in self.final_events if e.end >= cutoff]
