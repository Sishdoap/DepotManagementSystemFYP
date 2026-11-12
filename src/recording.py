"""Record the depot's live camera streams to hourly files (live-stream milestone R1).

The collaborator gave six live RTSP streams (2026-11-12; links with the
camera login in `.env`, gitignored). Ports 557-561 are the five cameras of
the 12 Aug footage, unchanged; 556 is new (near top-down over the barrier
and weighbridge). Everything downstream (tracking, code and plate reading,
vehicle type) runs on video files, so the streams are recorded
continuously: trucks are found afterwards, every frame is kept for
multi-frame voting, and a fixed rule can be re-run on the same footage.

Design decisions:

- **Stream copy, no re-encoding.** Packets are remuxed as they arrive
  (PyAV), so the files are bit-identical to what the camera sends and
  recording costs almost no CPU. This also keeps the door open for the
  live pipeline: the same connection can decode frames for live OCR, so
  each camera is opened once (the depot's NVR may limit viewers).
- **Matroska (.mkv)**, not .mp4: a file cut off by a crash or power loss
  is still readable; an unfinished .mp4 is not.
- **Hourly files aligned to the clock** (09:00, 10:00, ...), one folder
  per hour: `<root>/<YYYY-MM-DD>/<HH>00/cam<port>_<YYYY-MM-DD>_<HH-MM-SS>.mkv`.
  The pipeline scripts take one folder of simultaneous recordings, as
  with the 12 Aug footage, so an hour folder is a ready-made input. A file
  only starts on a keyframe (the first one after the hour), and its name
  carries the wall-clock time its first packet arrived — the clock offset
  to the burned-in overlay is measured afterwards
  (scripts/measure_clock_offsets.py), as before.
- **Reconnect on any error** with back-off; a reconnect starts a new file
  (stream timestamps restart), so one hour folder can hold two files for a
  camera after a drop. Every start, file close, disconnect and error goes
  to `recording_log.csv` in the day folder, so gaps are known.
- **The login never reaches a log.** URLs are masked (`mask_url`) before
  anything is printed or written; FFmpeg's own messages are not passed on.
"""

from __future__ import annotations

import csv
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

SEGMENT_S = 3600
FILE_FORMAT = "matroska"
FILE_SUFFIX = ".mkv"
LOG_NAME = "recording_log.csv"
LOG_COLUMNS = ["time", "camera", "event", "detail"]


def load_stream_urls(env_path) -> dict[str, str]:
    """{port: rtsp url} from `RTSP_CAM_<n>=rtsp://...:<port>/...` lines of a .env file."""
    urls = {}
    for line in Path(env_path).read_text().splitlines():
        m = re.match(r"\s*RTSP_CAM_\w+\s*=\s*['\"]?(rtsp://\S+?)['\"]?\s*$", line)
        if m:
            port = re.search(r"@?[\w.\-]+:(\d+)/", m.group(1))
            if port:
                urls[port.group(1)] = m.group(1)
    return urls


def mask_url(text: str) -> str:
    """Replace `user:password@` in any rtsp URL inside `text`."""
    return re.sub(r"(rtsp://)[^@/\s]+@", r"\1***@", str(text))


def recording_path(root, camera: str, started: datetime) -> Path:
    """Where the file for `camera` starting at `started` goes (see module docstring)."""
    day = started.strftime("%Y-%m-%d")
    name = f"cam{camera}_{day}_{started.strftime('%H-%M-%S')}{FILE_SUFFIX}"
    return Path(root) / day / started.strftime("%H00") / name


def next_boundary(now: datetime, segment_s: int = SEGMENT_S) -> datetime:
    """The next clock-aligned segment boundary strictly after `now`."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = (now - midnight).total_seconds()
    return midnight + timedelta(seconds=(int(elapsed // segment_s) + 1) * segment_s)


def in_hours(now: datetime, start_hour: int, end_hour: int) -> bool:
    """Is `now` within [start_hour, end_hour)? end_hour 24 means midnight."""
    return start_hour <= now.hour < end_hour


class RecordingLog:
    """Append-only CSV of recorder events, one file per day, shared by all cameras."""

    def __init__(self, root):
        self.root = Path(root)
        self._lock = threading.Lock()

    def write(self, camera: str, event: str, detail: str = "", now: Optional[datetime] = None):
        now = now or datetime.now()
        path = self.root / now.strftime("%Y-%m-%d") / LOG_NAME
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(LOG_COLUMNS)
                w.writerow([now.isoformat(timespec="seconds"), camera, event, mask_url(detail)])


@dataclass
class SegmentInfo:
    path: Path
    started: datetime
    packets: int = 0
    keyframes: int = 0


class StreamRecorder:
    """Copy one camera's stream into clock-aligned segment files until stopped.

    `source` is anything FFmpeg opens (an rtsp URL, or a file in tests).
    `clock` returns the current wall-clock time; tests replace it.
    """

    def __init__(self, camera: str, source: str, root, *, log: Optional[RecordingLog] = None,
                 segment_s: int = SEGMENT_S, clock: Callable[[], datetime] = datetime.now,
                 open_timeout_s: float = 15.0, read_timeout_s: float = 15.0):
        self.camera = camera
        self.source = source
        self.root = Path(root)
        self.log = log or RecordingLog(root)
        self.segment_s = segment_s
        self.clock = clock
        self.open_timeout_s = open_timeout_s
        self.read_timeout_s = read_timeout_s
        self.stop_event = threading.Event()
        self.segments: list[SegmentInfo] = []

    def _open_input(self):
        import av

        options = {}
        if self.source.startswith("rtsp://"):
            options = {"rtsp_transport": "tcp", "timeout": str(int(self.read_timeout_s * 1e6))}
        return av.open(self.source, options=options,
                       timeout=(self.open_timeout_s, self.read_timeout_s))

    def record_once(self, should_continue: Callable[[datetime], bool] = lambda now: True) -> int:
        """One connection: copy packets into segments until the stream ends, an
        error occurs, `stop()` is called or `should_continue(now)` turns false.
        Returns the number of packets written. Errors propagate to the caller.
        """
        import av

        inp = self._open_input()
        out = None
        seg: Optional[SegmentInfo] = None
        written = 0
        try:
            vin = inp.streams.video[0]
            vout = None
            boundary = None
            base = 0
            for pkt in inp.demux(vin):
                if self.stop_event.is_set():
                    break
                if pkt.dts is None or pkt.size == 0:
                    continue
                now = self.clock()
                if not should_continue(now):
                    break
                if out is not None and pkt.is_keyframe and now >= boundary:
                    self._close(out, seg)
                    out = None
                if out is None:
                    if not pkt.is_keyframe:
                        continue                      # a file starts on a keyframe
                    seg = SegmentInfo(recording_path(self.root, self.camera, now), now)
                    seg.path.parent.mkdir(parents=True, exist_ok=True)
                    out = av.open(str(seg.path), "w", format=FILE_FORMAT)
                    vout = out.add_stream_from_template(vin)
                    boundary = next_boundary(now, self.segment_s)
                    base = pkt.dts
                    self.segments.append(seg)
                    self.log.write(self.camera, "segment_open", seg.path.name, now)   # same clock reading as the name
                pkt.dts -= base
                if pkt.pts is not None:
                    pkt.pts -= base
                pkt.stream = vout
                out.mux(pkt)
                seg.packets += 1
                seg.keyframes += int(pkt.is_keyframe)
                written += 1
        finally:
            if out is not None:
                self._close(out, seg)
            inp.close()
        return written

    def _close(self, out, seg: SegmentInfo):
        out.close()
        self.log.write(self.camera, "segment_closed",
                       f"{seg.path.name} packets={seg.packets} keyframes={seg.keyframes}",
                       self.clock())

    def _log(self, event: str, detail: str = ""):
        self.log.write(self.camera, event, detail, self.clock())

    def run(self, should_continue: Callable[[datetime], bool] = lambda now: True,
            idle: Callable[[], None] = lambda: time.sleep(30),
            max_backoff_s: float = 60.0):
        """Record until `stop()`: reconnect after errors (back-off up to
        `max_backoff_s`), and wait via `idle` while `should_continue` is false
        (outside recording hours, low disk)."""
        backoff = 2.0
        self._log("start", mask_url(self.source))
        while not self.stop_event.is_set():
            if not should_continue(self.clock()):
                idle()
                continue
            try:
                n = self.record_once(should_continue)
                if n:
                    backoff = 2.0
                if not self.stop_event.is_set() and should_continue(self.clock()):
                    self._log("disconnect", f"stream ended after {n} packets")
            except Exception as e:                    # noqa: BLE001 — any failure means reconnect
                self._log("error", f"{type(e).__name__}: {mask_url(e)}")
            if self.stop_event.wait(backoff):
                break
            backoff = min(max_backoff_s, backoff * 2)
        self._log("stop")

    def stop(self):
        self.stop_event.set()
