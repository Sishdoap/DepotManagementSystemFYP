"""Tests for footage helpers (no video files or models needed)."""

import sys
from datetime import datetime, time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.footage import (
    CameraConfig, Detection, group_by_track, group_into_passes, keep_entry_lane_passes,
    load_camera_configs, load_tracks_csv, parse_overlay_time, parse_recording_name,
    pass_rejections, point_in_polygon, wall_clock,
)

NAME = "vlc-record-2026-08-12-09h15m42s-rtsp___175.138.143.18_558_Streaming_Channels_101-.mp4"
SQUARE = ((0, 0), (1000, 0), (1000, 1000), (0, 1000))


def det(t, area_side=100.0, conf=0.9, track_id=None, at=(0.0, 0.0)):
    x, y = at
    return Detection(t=t, frame=int(t * 20), label="truck", conf=conf,
                     box=(x, y, x + area_side, y + area_side), track_id=track_id)


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


class TestOverlayTime:
    def test_parses_clean_overlay(self):
        assert parse_overlay_time("08-12-2026 Wed 09:22:10") == time(9, 22, 10)

    def test_tolerates_ocr_spacing_and_separators(self):
        assert parse_overlay_time("08-12-2026Wed 09 :22.10") == time(9, 22, 10)

    def test_rejects_missing_or_impossible_time(self):
        assert parse_overlay_time("Camera 01") is None
        assert parse_overlay_time("08-12-2026 Wed 29:22:10") is None


class TestCameraConfig:
    def write(self, tmp_path, text):
        path = tmp_path / "cameras.yaml"
        path.write_text(text)
        return path

    def test_loads_roles_lane_and_offsets(self, tmp_path):
        configs = load_camera_configs(self.write(tmp_path, """
cameras:
  558:
    role: codes
    lane: [[0, 0], [10, 0], [10, 10]]
    min_box_px: 500
    clock_offset_s: [[2090, 8.4], [0, 6.6]]
  557:
    role: ignore
"""))
        cfg = configs["558"]
        assert cfg.role == "codes" and cfg.min_box_px == 500
        assert cfg.lane == ((0, 0), (10, 0), (10, 10))
        assert cfg.clock_offsets == ((0, 6.6), (2090, 8.4))    # sorted
        assert configs["557"].role == "ignore"

    def test_single_number_offset(self, tmp_path):
        configs = load_camera_configs(self.write(tmp_path, """
cameras:
  561: {role: cab, lane: [[0, 0], [1, 0], [1, 1]], clock_offset_s: 3.2}
"""))
        assert configs["561"].offset_at(0) == configs["561"].offset_at(5000) == 3.2

    def test_offset_is_a_step_function(self):
        cfg = CameraConfig("558", "codes", SQUARE, clock_offsets=((0, 6.6), (2090, 8.4)))
        assert cfg.offset_at(0) == 6.6
        assert cfg.offset_at(2089.9) == 6.6
        assert cfg.offset_at(2090) == 8.4

    def test_rejects_unknown_role(self, tmp_path):
        with pytest.raises(ValueError, match="role"):
            load_camera_configs(self.write(tmp_path, "cameras:\n  1: {role: parking}\n"))

    def test_active_camera_needs_lane(self, tmp_path):
        with pytest.raises(ValueError, match="lane"):
            load_camera_configs(self.write(tmp_path, "cameras:\n  1: {role: codes}\n"))

    def test_wall_clock_applies_offset(self):
        info = parse_recording_name(NAME)
        assert wall_clock(info, 10, 6.5) == datetime(2026, 8, 12, 9, 15, 58, 500000)


class TestLaneFilter:
    def test_point_in_polygon(self):
        assert point_in_polygon(500, 500, SQUARE)
        assert not point_in_polygon(1500, 500, SQUARE)
        # Non-convex: an L shape with the top-right quadrant cut away.
        ell = ((0, 0), (1000, 0), (1000, 500), (500, 500), (500, 1000), (0, 1000))
        assert point_in_polygon(250, 750, ell)
        assert not point_in_polygon(750, 750, ell)

    def test_group_by_track_keeps_overlapping_trucks_apart(self):
        dets = [det(0, track_id=1), det(0, track_id=2), det(0.5, track_id=1), det(0.5, track_id=2)]
        passes = group_by_track(dets + [det(0.2)])               # untracked one ignored
        assert [len(p.detections) for p in passes] == [2, 2]
        assert {p.detections[0].track_id for p in passes} == {1, 2}

    def test_keeps_large_trucks_in_lane(self):
        cfg = CameraConfig("558", "codes", SQUARE, min_box_px=300)
        in_lane = group_by_track([det(t, 400, track_id=1, at=(100, 100)) for t in (0, 0.5, 1)])
        assert keep_entry_lane_passes(in_lane, cfg) == in_lane

    def test_drops_out_of_lane_small_and_short_passes(self):
        cfg = CameraConfig("558", "codes", SQUARE, min_box_px=300)
        outside = [det(t, 400, track_id=1, at=(2000, 100)) for t in (0, 0.5, 1)]
        small = [det(t, 100, track_id=2, at=(100, 100)) for t in (0, 0.5, 1)]
        short = [det(t, 400, track_id=3, at=(100, 100)) for t in (0, 0.5)]
        assert keep_entry_lane_passes(group_by_track(outside + small + short), cfg) == []

    def test_lane_fraction_threshold(self):
        cfg = CameraConfig("558", "codes", SQUARE, min_box_px=0)
        # 2 of 4 centres in the lane: kept at 0.5, dropped at 0.75.
        dets = [det(0, 100, track_id=1, at=(100, 100)), det(1, 100, track_id=1, at=(100, 100)),
                det(2, 100, track_id=1, at=(2000, 100)), det(3, 100, track_id=1, at=(2000, 100))]
        passes = group_by_track(dets)
        assert len(keep_entry_lane_passes(passes, cfg, min_lane_fraction=0.5)) == 1
        assert keep_entry_lane_passes(passes, cfg, min_lane_fraction=0.75) == []

    def test_rejections_list_every_failed_check(self):
        cfg = CameraConfig("559", "backup", SQUARE, min_box_px=300, min_detections=3)
        p = group_by_track([det(0, 100, track_id=1, at=(2000, 100))])[0]
        reasons = pass_rejections(p, cfg)
        assert len(reasons) == 3
        assert reasons[0].startswith("only 1 frames") and reasons[1].startswith("small")
        assert reasons[2].startswith("outside entry lane")

    def test_kept_pass_has_no_rejections(self):
        cfg = CameraConfig("558", "codes", SQUARE, min_box_px=300)
        p = group_by_track([det(t, 400, track_id=1, at=(100, 100)) for t in (0, 0.5, 1)])[0]
        assert pass_rejections(p, cfg) == []


class TestTracksCsv:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "tracks.csv"
        path.write_text("camera,track_id,t,frame,label,conf,x1,y1,x2,y2\n"
                        "558,7,1.25,25,truck,0.8,10,20,110,220\n"
                        "561,3,0.5,10,bus,0.4,0,0,5,5\n")
        tracks = load_tracks_csv(path)
        d = tracks["558"][0]
        assert (d.track_id, d.t, d.frame, d.label, d.box) == (7, 1.25, 25, "truck", (10, 20, 110, 220))
        assert set(tracks) == {"558", "561"}
