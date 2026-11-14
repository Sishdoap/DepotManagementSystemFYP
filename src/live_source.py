"""Frames from the recorder's hourly files while they are being written (live mode).

Live mode does not open its own connection to the cameras: the NVR may
refuse extra viewers, and the user's snapshot recorder already holds one.
Instead it follows the files scripts/record_streams.py writes
(`<root>/<date>/<HH>00/cam<port>_<date>_<HH-MM-SS>.mkv`), the way
`tail -f` follows a log:

- `GrowingFile` is a read-only, non-seekable file object for PyAV. At the
  end of the data it waits for more instead of reporting end-of-file,
  until `finished()` says the file is complete. Being non-seekable, the
  Matroska demuxer reads it front to back like a network stream and never
  looks for the index that only exists once the file is closed.
- A file is complete when a newer file of the same camera exists: the
  recorder closes a segment before opening the next (src/recording.py),
  so by then every byte is on disk.
- `CameraFollower` walks a camera's files in time order and yields every
  `sample_every`-th decoded frame (20 fps / 5 = the 4 fps the batch
  tracker samples) with a time `t` in seconds since the first file's
  start, continuous across hourly files. The batch pipeline's `t` is the
  frame index / fps within one file; the same is used here within each
  file, plus the file's start offset from its name (1 s resolution — well
  inside the 2 s clock tolerance of event assembly).

Replaying finished recordings (`live=False`) uses the same code path, so
a recorded hour can stand in for the live streams (demo fallback, tests).

When a camera stops delivering (stream down), `stalled` turns true after
`stall_s` without new data, so the caller can stop waiting for that camera.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Optional

from .footage import RecordingInfo, VIDEO_SUFFIXES, parse_recording_name


class GrowingFile(io.RawIOBase):
    """A file that may still be growing; reads block until data or `finished()`.

    `stop()` (from another thread) makes the next wait return end-of-file.
    """

    def __init__(self, path, finished: Callable[[], bool], poll_s: float = 0.5,
                 on_wait: Optional[Callable[[float], None]] = None):
        super().__init__()
        self._f = open(path, "rb")
        self._finished = finished
        self._poll_s = poll_s
        self._on_wait = on_wait
        self._stopped = False
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def stop(self) -> None:
        self._stopped = True

    def readinto(self, b) -> int:
        waited = 0.0
        while True:
            n = self._f.readinto(b)
            if n:
                self.bytes_read += n
                return n
            if self._stopped:
                return 0
            if self._finished():
                n = self._f.readinto(b)          # anything written just before it was closed
                self.bytes_read += n or 0
                return n or 0
            time.sleep(self._poll_s)
            waited += self._poll_s
            if self._on_wait:
                self._on_wait(waited)

    def close(self) -> None:
        self._f.close()
        super().close()


def camera_files(root, camera: str) -> list[tuple[RecordingInfo, Path]]:
    """Every recording of `camera` under `root` (any depth), oldest first."""
    out = []
    for p in Path(root).rglob(f"cam{camera}_*"):
        info = parse_recording_name(p.name)
        if info and info.camera == camera and p.suffix.lower() in VIDEO_SUFFIXES:
            out.append((info, p))
    return sorted(out, key=lambda ip: ip[0].started_at)


@dataclass
class SampledFrame:
    camera: str
    t: float                      # seconds since the follower's reference start
    frame: int                    # running frame index across files
    image: object                 # BGR numpy array
    file: Path


class CameraFollower:
    """Sampled frames of one camera, across its files, as they are written."""

    def __init__(self, root, camera: str, *, live: bool = True, sample_every: int = 5,
                 fps: float = 20.0, start_after: Optional[datetime] = None,
                 files: Optional[list[Path]] = None, poll_s: float = 0.5, stall_s: float = 30.0,
                 decode_threads: int = 4):
        self.root, self.camera, self.live = Path(root), camera, live
        self.sample_every, self.fps = sample_every, fps
        self.start_after = start_after
        self.files = files
        self.poll_s, self.stall_s = poll_s, stall_s
        self.decode_threads = decode_threads
        self.reference: Optional[RecordingInfo] = None
        self.stalled = False
        self._current: Optional[GrowingFile] = None
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True
        if self._current is not None:
            self._current.stop()

    def _list(self) -> list[tuple[RecordingInfo, Path]]:
        if self.files is not None:
            return sorted(((parse_recording_name(p.name), Path(p)) for p in self.files),
                          key=lambda ip: ip[0].started_at)
        return camera_files(self.root, self.camera)

    def _first(self) -> Optional[tuple[RecordingInfo, Path]]:
        files = self._list()
        if not files:
            return None
        if self.live and self.start_after is None:
            return files[-1]                     # the file being written now
        if self.start_after is not None:
            later = [ip for ip in files if ip[0].started_at >= self.start_after]
            earlier = [ip for ip in files if ip[0].started_at < self.start_after]
            return (earlier[-1] if earlier else None) or (later[0] if later else None)
        return files[0]

    def _newer_exists(self, info: RecordingInfo) -> bool:
        return any(i.started_at > info.started_at for i, _ in self._list())

    def _on_wait(self, waited: float) -> None:
        self.stalled = waited >= self.stall_s

    def frames(self) -> Iterator[SampledFrame]:
        """Yield sampled frames until stopped (live) or the last file ends (replay)."""
        import av

        while not self._stopped:
            first = self._first()
            if first is not None:
                break
            if not self.live:
                return
            time.sleep(self.poll_s)
        else:
            return
        info, path = first
        self.reference = info
        running = 0
        while not self._stopped:
            finished = (lambda: True) if not self.live else (lambda i=info: self._newer_exists(i))
            gf = GrowingFile(path, finished, poll_s=self.poll_s, on_wait=self._on_wait)
            self._current = gf
            offset = (info.started_at - self.reference.started_at).total_seconds()
            try:
                with av.open(gf, mode="r", format="matroska" if path.suffix == ".mkv" else None) as c:
                    stream = c.streams.video[0]
                    stream.thread_type = "AUTO"
                    stream.codec_context.thread_count = self.decode_threads
                    idx = 0
                    for packet in c.demux(stream):
                        if self._stopped:
                            break
                        for frame in packet.decode():
                            self.stalled = False
                            if idx % self.sample_every == 0:
                                yield SampledFrame(self.camera, offset + idx / self.fps, running,
                                                   frame.to_ndarray(format="bgr24"), path)
                            idx += 1
                            running += 1
            except (av.error.InvalidDataError, av.error.EOFError):
                pass                              # truncated file (recorder killed): next file
            finally:
                gf.close()
                self._current = None
            # next file of this camera
            while not self._stopped:
                later = [ip for ip in self._list() if ip[0].started_at > info.started_at]
                if later:
                    info, path = later[0]
                    break
                if not self.live:
                    return
                time.sleep(self.poll_s)
