"""Tests for the live-stream recorder (src/recording.py)."""

import csv
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.recording import (
    LOG_NAME, RecordingLog, StreamRecorder, in_hours, load_stream_urls, mask_url, next_boundary,
    recording_path, repair_recording,
)

av = pytest.importorskip("av")


def test_load_stream_urls(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# cameras\n"
                   "RTSP_CAM_157=rtsp://user:p%40ss@10.0.0.1:557/Streaming/Channels/101\n"
                   "RTSP_CAM_161='rtsp://user:pw@10.0.0.1:561/Streaming/Channels/101'\n"
                   "OTHER=x\n")
    urls = load_stream_urls(env)
    assert sorted(urls) == ["557", "561"]
    assert urls["561"] == "rtsp://user:pw@10.0.0.1:561/Streaming/Channels/101"


def test_mask_url_hides_login():
    text = "error opening rtsp://admin:secret@1.2.3.4:558/Streaming/Channels/101 (timeout)"
    masked = mask_url(text)
    assert "secret" not in masked and "admin" not in masked
    assert "rtsp://***@1.2.3.4:558/" in masked
    assert mask_url("rtsp://1.2.3.4:558/x") == "rtsp://1.2.3.4:558/x"


def test_recording_path_layout():
    p = recording_path("/rec", "561", datetime(2026, 11, 13, 9, 0, 2))
    assert p == Path("/rec/2026-11-13/0900/cam561_2026-11-13_09-00-02.mkv")


def test_next_boundary_is_clock_aligned():
    assert next_boundary(datetime(2026, 11, 13, 9, 59, 59)) == datetime(2026, 11, 13, 10, 0)
    assert next_boundary(datetime(2026, 11, 13, 10, 0, 0)) == datetime(2026, 11, 13, 11, 0)
    assert next_boundary(datetime(2026, 11, 13, 23, 30)) == datetime(2026, 11, 14, 0, 0)
    assert next_boundary(datetime(2026, 11, 13, 9, 7), 600) == datetime(2026, 11, 13, 9, 10)


def test_in_hours():
    assert in_hours(datetime(2026, 11, 13, 7, 0), 7, 19)
    assert not in_hours(datetime(2026, 11, 13, 19, 0), 7, 19)
    assert in_hours(datetime(2026, 11, 13, 23, 59), 0, 24)


def test_log_masks_and_appends(tmp_path):
    log = RecordingLog(tmp_path)
    now = datetime(2026, 11, 13, 9, 0)
    log.write("558", "error", "cannot open rtsp://u:pw@h:558/x", now)
    log.write("558", "start", "", now)
    rows = list(csv.DictReader(open(tmp_path / "2026-11-13" / LOG_NAME)))
    assert [r["event"] for r in rows] == ["error", "start"]
    assert "pw" not in rows[0]["detail"]


def make_video(path, n_frames=60, gop=10, fps=20):
    """A small mpeg4 video with a keyframe every `gop` frames."""
    with av.open(str(path), "w") as out:
        s = out.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        s.codec_context.gop_size = gop
        s.codec_context.time_base = Fraction(1, fps)
        for i in range(n_frames):
            img = np.full((48, 64, 3), (i * 4) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for p in s.encode(frame):
                out.mux(p)
        for p in s.encode():
            out.mux(p)


def frames_in(path):
    with av.open(str(path)) as f:
        return sum(1 for _ in f.decode(video=0))


class FakeClock:
    def __init__(self, start, step_s):
        self.now, self.step = start, timedelta(seconds=step_s)

    def __call__(self):
        t = self.now
        self.now += self.step
        return t


def keyframe_indices(path):
    with av.open(str(path)) as f:
        return [i for i, p in enumerate(p for p in f.demux(video=0) if p.size) if p.is_keyframe]


def test_recorder_splits_at_boundary_on_keyframe(tmp_path):
    src = tmp_path / "src.avi"
    make_video(src)
    keys = keyframe_indices(src)
    split = keys[-3]                      # a keyframe well inside the clip (the encoder
                                          # makes every early frame a keyframe)
    # 0.05 s per packet, the hour passes exactly at that keyframe.
    start = datetime(2026, 11, 13, 10, 0) - timedelta(seconds=0.05 * split)
    rec = StreamRecorder("561", str(src), tmp_path / "rec", clock=FakeClock(start, 0.05))
    n = rec.record_once()
    assert n == 60
    assert [s.packets for s in rec.segments] == [split, 60 - split]
    first, second = (s.path for s in rec.segments)
    assert first.parent.name == "0900" and second.parent.name == "1000"
    assert second.name == "cam561_2026-11-13_10-00-00.mkv"
    assert frames_in(first) + frames_in(second) == 60
    log = list(csv.DictReader(open(tmp_path / "rec" / "2026-11-13" / LOG_NAME)))
    assert [r["event"] for r in log].count("segment_closed") == 2


def test_recorder_waits_for_keyframe_after_boundary(tmp_path):
    src = tmp_path / "src.avi"
    make_video(src)
    keys = keyframe_indices(src)
    k1, k2 = next((a, b) for a, b in zip(keys, keys[1:]) if b - a > 2)
    passes = k1 + 1                       # the hour passes just after a keyframe ...
    start = datetime(2026, 11, 13, 10, 0) - timedelta(seconds=0.05 * passes)
    rec = StreamRecorder("558", str(src), tmp_path / "rec", clock=FakeClock(start, 0.05))
    rec.record_once()
    assert [s.packets for s in rec.segments] == [k2, 60 - k2]   # ... split waits for the next


def test_repair_cut_off_recording(tmp_path):
    import cv2

    src = tmp_path / "src.avi"
    make_video(src)
    rec = StreamRecorder("558", str(src), tmp_path / "rec",
                         clock=FakeClock(datetime(2026, 11, 13, 7, 0), 0.05))
    rec.record_once()
    good = rec.segments[0].path
    cut = tmp_path / "cut.mkv"
    data = good.read_bytes()
    cut.write_bytes(data[: int(len(data) * 0.8)])       # writer killed: no index, no duration
    fixed = tmp_path / "fixed.mkv"
    n = repair_recording(cut, fixed)
    assert 30 < n < 60
    cap = cv2.VideoCapture(str(fixed))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pytest.approx(n, abs=2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    assert cap.read()[0]


def test_recorder_stops_when_told(tmp_path):
    src = tmp_path / "src.avi"
    make_video(src)
    clock = FakeClock(datetime(2026, 11, 13, 9, 0), 0.05)
    rec = StreamRecorder("558", str(src), tmp_path / "rec", clock=clock)
    stop_at = datetime(2026, 11, 13, 9, 0, 1)            # after 20 packets
    rec.record_once(should_continue=lambda now: now < stop_at)
    assert [s.packets for s in rec.segments] == [20]
    assert frames_in(rec.segments[0].path) == 20
