"""Tests for footage helpers (no video files or models needed)."""

import sys
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.footage import Detection, group_into_passes, parse_recording_name, wall_clock

NAME = "vlc-record-2026-08-12-09h15m42s-rtsp___175.138.143.18_558_Streaming_Channels_101-.mp4"


def det(t, area_side=100.0, conf=0.9):
    return Detection(t=t, frame=int(t * 20), label="truck", conf=conf,
                     box=(0.0, 0.0, area_side, area_side))


class TestRecordingName:
    def test_parses_camera_and_start(self):
        info = parse_recording_name(NAME)
        assert info.camera == "558"
        assert info.started_at == datetime(2026, 8, 12, 9, 15, 42)

    def test_accepts_full_path(self):
        assert parse_recording_name("/x/videos/" + NAME).camera == "558"

    def test_rejects_other_names(self):
        assert parse_recording_name("clip.mp4") is None

    def test_wall_clock(self):
        info = parse_recording_name(NAME)
        assert wall_clock(info, 90.5) == datetime(2026, 8, 12, 9, 17, 12, 500000)


class TestGrouping:
    def test_gap_splits_passes(self):
        passes = group_into_passes([det(0), det(0.5), det(1.0), det(20), det(20.5)])
        assert [(p.start, p.end) for p in passes] == [(0, 1.0), (20, 20.5)]

    def test_small_gap_keeps_one_pass(self):
        passes = group_into_passes([det(0), det(3), det(6)], max_gap_seconds=5)
        assert len(passes) == 1

    def test_unsorted_input(self):
        passes = group_into_passes([det(20.5), det(0), det(20), det(0.5)])
        assert [p.start for p in passes] == [0, 20]

    def test_single_frame_blips_dropped(self):
        passes = group_into_passes([det(0), det(30), det(30.5)])
        assert [p.start for p in passes] == [30]

    def test_best_is_largest_detection(self):
        p = group_into_passes([det(0, 50), det(0.5, 300), det(1.0, 120)])[0]
        assert p.best.t == 0.5

    def test_empty(self):
        assert group_into_passes([]) == []
