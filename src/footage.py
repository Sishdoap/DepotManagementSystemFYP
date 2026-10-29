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
from datetime import datetime, timedelta
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

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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


def wall_clock(info: RecordingInfo, seconds: float) -> datetime:
    """Wall-clock time of a moment `seconds` into the recording."""
    return info.started_at + timedelta(seconds=seconds)
