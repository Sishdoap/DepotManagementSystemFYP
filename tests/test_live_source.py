"""Tests for following recordings while they are written (src/live_source.py)."""

import shutil
import sys
import threading
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import av
import numpy as np

from src.live_source import CameraFollower, GrowingFile, camera_files


def make_mkv(path, n_frames=40, fps=20):
    """A small Matroska video, keyframe every 10 frames."""
    with av.open(str(path), "w", format="matroska") as out:
        s = out.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.gop_size = 10
        s.codec_context.time_base = Fraction(1, fps)
        for i in range(n_frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 5 % 256, np.uint8),
                                               format="rgb24")
            frame.pts = i
            for p in s.encode(frame):
                out.mux(p)
        for p in s.encode():
            out.mux(p)


def count_frames(fileobj):
    with av.open(fileobj, format="matroska") as c:
        return sum(1 for _ in c.decode(video=0))


def write_slowly(src, dst, chunk=2000, delay=0.005, done=None):
    data = Path(src).read_bytes()
    with open(dst, "wb") as f:
        for i in range(0, len(data), chunk):
            f.write(data[i:i + chunk])
            f.flush()
            time.sleep(delay)
    if done is not None:
        done.set()


def test_growing_file_reads_until_finished(tmp_path):
    src, dst = tmp_path / "src.mkv", tmp_path / "dst.mkv"
    make_mkv(src)
    expected = count_frames(open(src, "rb"))
    dst.write_bytes(b"")
    done = threading.Event()
    writer = threading.Thread(target=write_slowly, args=(src, dst), kwargs={"done": done})
    writer.start()
    gf = GrowingFile(dst, done.is_set, poll_s=0.01)
    try:
        assert not gf.seekable()
        assert count_frames(gf) == expected
    finally:
        gf.close()
        writer.join()


def test_growing_file_stop_returns_eof(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    gf = GrowingFile(p, lambda: False, poll_s=0.01)
    assert gf.read(10) == b"abc"
    threading.Timer(0.05, gf.stop).start()
    assert gf.read(10) == b""                     # waits, then the stop ends it
    gf.close()


def names(tmp_path, *hours):
    out = []
    for h in hours:
        d = tmp_path / "2026-11-13" / f"{h:02d}00"
        d.mkdir(parents=True, exist_ok=True)
        out.append(d / f"cam557_2026-11-13_{h:02d}-00-00.mkv")
    return out


def test_camera_files_sorted_and_filtered(tmp_path):
    a, b = names(tmp_path, 12, 11)
    for p in (a, b):
        p.write_bytes(b"")
    (tmp_path / "2026-11-13" / "1100" / "cam558_2026-11-13_11-00-00.mkv").write_bytes(b"")
    (tmp_path / "2026-11-13" / "1100" / "notes.txt").write_bytes(b"")
    assert [p for _, p in camera_files(tmp_path, "557")] == [b, a]


def test_replay_samples_and_continuous_time(tmp_path):
    a, b = names(tmp_path, 11, 12)
    make_mkv(a, n_frames=40)
    make_mkv(b, n_frames=40)
    f = CameraFollower(tmp_path, "557", live=False, sample_every=5)
    got = list(f.frames())
    assert len(got) == 16                                   # 8 sampled frames per file
    assert [round(s.t, 2) for s in got[:3]] == [0.0, 0.25, 0.5]
    assert got[8].t == 3600.0 and got[8].file == b         # second file: one hour later
    assert got[8].frame == 40                                # running frame index
    assert got[0].image.shape == (48, 64, 3)
    assert f.reference.started_at == datetime(2026, 11, 13, 11, 0, 0)


def test_live_follows_into_next_file(tmp_path):
    a, b = names(tmp_path, 11, 12)
    src = tmp_path / "src.mkv"
    make_mkv(src, n_frames=40)
    shutil.copy(src, a)                     # 11:00 is complete; 12:00 appears later
    f = CameraFollower(tmp_path, "557", live=True, start_after=datetime(2026, 11, 13, 11, 0),
                       sample_every=5, poll_s=0.01)
    got = []

    def consume():
        for s in f.frames():
            got.append(s)
            if len(got) == 16:
                f.stop()

    reader = threading.Thread(target=consume, daemon=True)   # a failed assert must not hang pytest
    reader.start()
    deadline = time.time() + 5
    while len(got) < 4 and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    # 11:00 read; waiting for the next file
    assert 4 <= len(got) <= 8         # (a threaded decoder may hold a few back)
    writer = threading.Thread(target=write_slowly, args=(src, b))
    writer.start()
    writer.join()
    # 12:00 is finished (and its held-back frames released) once 13:00 appears
    names(tmp_path, 13)[0].write_bytes(b"")
    deadline = time.time() + 5
    while len(got) < 16 and time.time() < deadline:
        time.sleep(0.05)
    f.stop()
    reader.join(timeout=5)
    assert len(got) == 16 and got[8].t == 3600.0


def test_stalled_after_no_data(tmp_path):
    (a,) = names(tmp_path, 11)
    a.write_bytes(b"")
    f = CameraFollower(tmp_path, "557", live=True, poll_s=0.01, stall_s=0.05)
    t = threading.Thread(target=lambda: list(f.frames()), daemon=True)
    t.start()
    deadline = time.time() + 2
    while not f.stalled and time.time() < deadline:
        time.sleep(0.01)
    assert f.stalled
    f.stop()
    t.join(timeout=5)
    assert not t.is_alive()


def test_skip_starts_at_a_keyframe_without_decoding_before(tmp_path):
    (a,) = names(tmp_path, 11)
    make_mkv(a, n_frames=40)
    with av.open(str(a)) as c:
        s = c.streams.video[0]
        keys = [float(p.pts * s.time_base) for p in c.demux(s) if p.size and p.is_keyframe]
    first_key = min(k for k in keys if k >= 0.6)
    f = CameraFollower(tmp_path, "557", live=False, sample_every=5, skip_s=0.6)
    got = list(f.frames())
    start = round(first_key * 20)
    assert got[0].frame == next(i for i in range(start, 40) if i % 5 == 0)   # sampled frames only
    assert [s.t for s in got] == [i / 20 for i in range(start, 40) if i % 5 == 0]
