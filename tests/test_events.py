"""Tests for cross-camera event assembly (no video files or models needed)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.events import CameraPass, assemble_events, camera_chain, extract_events
from src.footage import CameraConfig, Detection, RecordingInfo

T0 = datetime(2026, 8, 12, 9, 0, 0)
CHAIN = ["561", "558", "559"]


def cp(camera, start, end=None, track_id=None):
    """A pass from `start` to `end` seconds after T0 (end defaults to start + 2)."""
    end = start + 2 if end is None else end
    return CameraPass(camera, track_id if track_id is not None else int(start),
                      T0 + timedelta(seconds=start), T0 + timedelta(seconds=end))


def shape(events):
    """[(camera, start second), ...] per event, for compact assertions."""
    return [[(p.camera, int((p.start - T0).total_seconds())) for p in e.passes] for e in events]


class TestAssembly:
    def test_one_truck_across_three_cameras(self):
        events = assemble_events([cp("561", 0), cp("558", 5), cp("559", 4)], CHAIN)
        assert len(events) == 1
        assert events[0].cameras == ["561", "559", "558"]      # first-seen order
        assert events[0].head_seen

    def test_entry_time_is_first_sighting(self):
        # A later camera may see the truck slightly before the head (clock tolerance).
        events = assemble_events([cp("561", 10), cp("558", 9)], CHAIN)
        assert len(events) == 1
        assert events[0].start == T0 + timedelta(seconds=9)

    def test_back_to_back_trucks_stay_apart(self):
        # Two trucks 31 s apart (the closest pair in the footage).
        passes = [cp("561", 0), cp("558", 4), cp("558", 20),
                  cp("561", 31), cp("558", 35), cp("558", 52)]
        assert shape(assemble_events(passes, CHAIN)) == [
            [("561", 0), ("558", 4), ("558", 20)],
            [("561", 31), ("558", 35), ("558", 52)],
        ]

    def test_cab_and_container_fragments_join_one_event(self):
        # 558 sees the cab, then the container doors ~20 s later.
        events = assemble_events([cp("561", 0), cp("558", 8, 9), cp("558", 28, 41)], CHAIN)
        assert len(events) == 1 and len(events[0].on("558")) == 2

    def test_continuing_fragment_may_exceed_max_lag(self):
        # Truck stops under 558: fragments keep coming past max_lag_s.
        passes = [cp("561", 0), cp("558", 30, 40), cp("558", 55, 60)]
        assert len(assemble_events(passes, CHAIN, max_lag_s=45)) == 1

    def test_head_camera_miss_opens_own_event(self):
        events = assemble_events([cp("561", 0), cp("558", 4), cp("558", 120)], CHAIN)
        assert len(events) == 2
        assert events[1].head_seen is False and events[1].cameras == ["558"]

    def test_late_pass_beyond_max_lag_opens_new_event(self):
        events = assemble_events([cp("561", 0), cp("559", 60)], CHAIN, max_lag_s=45)
        assert [e.head_seen for e in events] == [True, False]

    def test_fragment_after_long_gap_is_a_new_truck(self):
        # 558 fragments 30 s apart (> merge_gap_s) within max_lag: second is a new truck
        # the head camera missed, not part of the first event.
        passes = [cp("561", 0), cp("558", 2, 4), cp("558", 34)]
        events = assemble_events(passes, CHAIN, max_lag_s=45, merge_gap_s=25)
        assert len(events) == 2 and events[1].head_seen is False

    def test_pass_joins_most_recent_truck_only(self):
        # No overtaking: once truck B has entered, a 558 pass belongs to B, not A.
        passes = [cp("561", 0), cp("561", 33), cp("558", 36)]
        events = assemble_events(passes, CHAIN)
        assert events[0].cameras == ["561"] and events[1].cameras == ["561", "558"]

    def test_broken_head_track_merges(self):
        events = assemble_events([cp("561", 0, 2), cp("561", 4, 6)], CHAIN, head_merge_gap_s=5)
        assert len(events) == 1

    def test_cameras_outside_chain_ignored(self):
        events = assemble_events([cp("561", 0), cp("557", 1)], CHAIN)
        assert events[0].cameras == ["561"]

    def test_input_order_does_not_matter(self):
        passes = [cp("561", 0), cp("558", 5), cp("561", 40), cp("558", 44), cp("559", 46)]
        assert shape(assemble_events(passes, CHAIN)) == shape(assemble_events(passes[::-1], CHAIN))

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError):
            assemble_events([cp("561", 0)], [])

    def test_no_passes(self):
        assert assemble_events([], CHAIN) == []


LANE = ((0, 0), (1000, 0), (1000, 1000), (0, 1000))


def tracked(t, track_id, at=(100, 100), side=400):
    x, y = at
    return Detection(t, int(t * 20), "truck", 0.9, (x, y, x + side, y + side), track_id)


class TestExtractEvents:
    configs = {
        "558": CameraConfig("558", "codes", LANE, min_box_px=300, clock_offsets=((0, 5.0),)),
        "561": CameraConfig("561", "cab", LANE, min_box_px=300),
        "557": CameraConfig("557", "ignore"),
    }
    recordings = {c: RecordingInfo(c, T0) for c in ("558", "561", "557")}

    def test_chain_follows_roles(self):
        assert camera_chain(self.configs) == ["561", "558"]

    def test_filters_offsets_and_assembles(self):
        tracks = {
            "561": [tracked(t, 1) for t in (10, 10.5, 11)],
            # 558 video time 8 s + 5 s clock offset = 13 s wall: 3 s after the cab.
            "558": [tracked(t, 4) for t in (8, 8.5, 9)]
                   + [tracked(t, 5, at=(5000, 100)) for t in (8, 8.5, 9)],   # outbound lane
        }
        ex = extract_events(tracks, self.configs, self.recordings)
        assert len(ex.events) == 1
        e = ex.events[0]
        assert e.cameras == ["561", "558"]
        assert e.on("558")[0].start == T0 + timedelta(seconds=13)
        assert set(ex.detections) == {("561", 1), ("558", 4)}
        assert [(r["camera"], r["track_id"]) for r in ex.rejected] == [("558", 5)]
        assert "outside entry lane" in ex.rejected[0]["reason"]
