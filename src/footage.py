"""Helpers for working with the collaborator's depot CCTV footage.

The footage is a set of simultaneous recordings, one file per camera, named
by VLC after the RTSP stream they were recorded from, e.g.

    vlc-record-2026-08-12-09h15m42s-rtsp___175.138.143.18_558_Streaming_Channels_101-.mp4

The date/time in the name is when the recording started, and the number
before "_Streaming" is the camera's RTSP port, used here as the camera id.

This module stays free of model code so it can be unit-tested; detection
itself lives in scripts/scan_trucks.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional


_VLC_NAME_RE = re.compile(
    r"vlc-record-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s-rtsp_+[\d.]+_(\d+)_"
)


@dataclass(frozen=True)
class RecordingInfo:
    """Camera id and recording start time parsed from a VLC filename."""
    camera: str               # RTSP port, e.g. "558"
    started_at: datetime      # wall-clock time of the first frame


def parse_recording_name(name: str) -> Optional[RecordingInfo]:
    """Parse a VLC recording filename; None if it doesn't match."""
    m = _VLC_NAME_RE.search(Path(name).name)
    if not m:
        return None
    y, mo, d, h, mi, s, port = m.groups()
    return RecordingInfo(port, datetime(int(y), int(mo), int(d), int(h), int(mi), int(s)))


@dataclass(frozen=True)
class Detection:
    """One vehicle detection in one sampled frame."""
    t: float                  # seconds from the start of the video
    frame: int                # frame index in the video
    label: str                # detector class, e.g. "truck"
    conf: float
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 in full-frame pixels
    track_id: Optional[int] = None           # tracker id; None for untracked scans

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def longest_side(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


@dataclass
class TruckPass:
    """A run of sampled frames in which a truck is visible on one camera."""
    detections: list[Detection] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.detections[0].t

    @property
    def end(self) -> float:
        return self.detections[-1].t

    @property
    def best(self) -> Detection:
        """The detection where the truck appears largest (best for reading text)."""
        return max(self.detections, key=lambda d: (d.area, d.conf))


def group_into_passes(
    detections: Iterable[Detection],
    *,
    max_gap_seconds: float = 5.0,
    min_detections: int = 2,
) -> list[TruckPass]:
    """Group per-frame detections from one camera into truck passes.

    Detections are sorted by time; a new pass starts whenever more than
    `max_gap_seconds` pass without a detection. This assumes one truck in
    view at a time, which holds at a single-lane gate. Passes with fewer
    than `min_detections` sampled frames are dropped as likely false
    positives (e.g. one frame where a parked van looks like a truck).
    """
    passes: list[TruckPass] = []
    current: Optional[TruckPass] = None
    for det in sorted(detections, key=lambda d: d.t):
        if current is None or det.t - current.end > max_gap_seconds:
            current = TruckPass()
            passes.append(current)
        current.detections.append(det)
    return [p for p in passes if len(p.detections) >= min_detections]


def group_by_track(detections: Iterable[Detection]) -> list[TruckPass]:
    """One pass per tracker id, sorted by start time.

    Replaces gap-based grouping for tracked scans: two trucks in view at
    once, or following each other closely, keep separate ids.
    Untracked detections (track_id None) are ignored.
    """
    by_id: dict[int, TruckPass] = {}
    for det in sorted(detections, key=lambda d: d.t):
        if det.track_id is not None:
            by_id.setdefault(det.track_id, TruckPass()).detections.append(det)
    return sorted(by_id.values(), key=lambda p: p.start)


def keep_entry_lane_passes(
    passes: Iterable[TruckPass],
    config: "CameraConfig",
    *,
    min_lane_fraction: float = 0.5,
) -> list[TruckPass]:
    """Drop passes that are not trucks in this camera's entry lane.

    A pass is kept when at least `min_lane_fraction` of its box centres
    are inside the lane polygon (parked trucks, the outbound lane and yard
    traffic fail this), its largest box reaches `config.min_box_px` (far-
    away vehicles fail this), and it has `config.min_detections` sampled
    frames (flickers fail this).
    """
    kept = []
    for p in passes:
        if len(p.detections) < config.min_detections:
            continue
        if max(d.longest_side for d in p.detections) < config.min_box_px:
            continue
        in_lane = sum(config.in_lane(d) for d in p.detections)
        if in_lane < min_lane_fraction * len(p.detections):
            continue
        kept.append(p)
    return kept


def wall_clock(info: RecordingInfo, seconds: float, offset_seconds: float = 0.0) -> datetime:
    """Wall-clock time of a moment `seconds` into the recording.

    `offset_seconds` corrects the filename start time (when VLC began
    recording) to the time actually shown in the video; see
    CameraConfig.clock_offsets.
    """
    return info.started_at + timedelta(seconds=seconds + offset_seconds)


# The cameras burn "MM-DD-YYYY Day HH:MM:SS" into the top-left corner, e.g.
# "08-12-2026 Wed 09:22:10" on 12 Aug 2026. Only the time is used: the date
# comes from the filename, and OCR often merges or drops the separators.
_OVERLAY_TIME_RE = re.compile(r"(\d{2})[:.;](\d{2})[:.;](\d{2})(?!\d)")


def parse_overlay_time(text: str) -> Optional[time]:
    """Time of day from OCR of the burned-in clock; None if not found."""
    m = _OVERLAY_TIME_RE.search(text.replace(" ", ""))
    if not m:
        return None
    h, mi, s = (int(g) for g in m.groups())
    if h > 23 or mi > 59 or s > 59:
        return None
    return time(h, mi, s)

# --- Camera configuration -------------------------------------------------

CAMERA_ROLES = ("cab", "codes", "backup", "ignore")


@dataclass(frozen=True)
class CameraConfig:
    """How one camera is used by the event extractor (config/cameras.yaml).

    role           "cab" (cab front: make, plate, fleet number), "codes"
                   (container doors/roof: ISO 6346 code, size/type),
                   "backup" (second view of the codes), or "ignore".
    lane           polygon (full-frame pixels) around the entry lane. A
                   track counts only if most of its box centres fall inside,
                   which drops parked vehicles, the outbound lane and yard
                   traffic.
    min_box_px     a track is kept only if its largest box has a side at
                   least this long; small boxes are vehicles far away.
    min_detections a track needs this many sampled frames. 559 sees trucks
                   only briefly and close up, so it needs a lower value
                   (with a higher min_box_px) than the other cameras.
    clock_offsets  seconds added to the filename start time so that video
                   time matches the clock burned into the frame, as a step
                   function: ((from video second, offset), ...), sorted.
                   A step function rather than offset + drift because the
                   measured offsets are flat except for one jump, when the
                   streams dropped frames (scripts/measure_clock_offsets.py).
    """
    camera: str
    role: str
    lane: tuple[tuple[float, float], ...] = ()
    min_box_px: float = 0.0
    min_detections: int = 3
    clock_offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),)

    def offset_at(self, seconds: float) -> float:
        """Clock offset in force `seconds` into the video."""
        offset = self.clock_offsets[0][1]
        for start, value in self.clock_offsets:
            if seconds >= start:
                offset = value
        return offset

    def in_lane(self, det: "Detection") -> bool:
        """Is the centre of the detection's box inside the entry lane?"""
        return point_in_polygon(*det.center, self.lane)


def load_camera_configs(path) -> dict[str, CameraConfig]:
    """Read config/cameras.yaml into {camera id: CameraConfig}."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    configs = {}
    for camera, spec in (raw.get("cameras") or {}).items():
        camera = str(camera)
        role = spec.get("role", "ignore")
        if role not in CAMERA_ROLES:
            raise ValueError(f"camera {camera}: unknown role {role!r} (expected one of {CAMERA_ROLES})")
        lane = tuple((float(x), float(y)) for x, y in spec.get("lane") or ())
        if role != "ignore" and len(lane) < 3:
            raise ValueError(f"camera {camera}: lane polygon needs at least 3 points")
        configs[camera] = CameraConfig(
            camera=camera, role=role, lane=lane,
            min_box_px=float(spec.get("min_box_px", 0)),
            min_detections=int(spec.get("min_detections", 3)),
            clock_offsets=_parse_clock_offsets(spec.get("clock_offset_s", 0)),
        )
    return configs


def _parse_clock_offsets(value) -> tuple[tuple[float, float], ...]:
    """A single number, or a list of [from video second, offset] steps."""
    if isinstance(value, (int, float)):
        return ((0.0, float(value)),)
    steps = tuple(sorted((float(start), float(offset)) for start, offset in value))
    if not steps:
        raise ValueError("clock_offset_s list is empty")
    return steps


def point_in_polygon(x: float, y: float, polygon) -> bool:
    """Ray-casting test: is (x, y) inside the polygon (list of (x, y) vertices)?"""
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside
