"""Tests for live event assembly: same events as the batch pipeline, finalised in time."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.events import extract_events
from src.footage import (
    CameraConfig, Detection, RecordingInfo, find_recordings, load_camera_configs, load_tracks_csv,
    parse_recording_name,
)
from src.live_events import LiveEventAssembler, event_id

START = datetime(2026, 11, 13, 9, 0, 0)
LANE = ((0, 0), (5000, 0), (5000, 1440), (0, 1440))


def cfg(cam, role, **kw):
    return CameraConfig(camera=cam, role=role, lane=LANE, min_box_px=0, **kw)


CONFIGS = {"561": cfg("561", "cab"), "558": cfg("558", "codes")}
RECS = {c: RecordingInfo(c, START) for c in CONFIGS}


def track(cam, tid, t0, t1, step=0.25):
    n = int((t1 - t0) / step) + 1
    return [Detection(t=t0 + i * step, frame=int((t0 + i * step) * 20), label="truck", conf=0.9,
                      box=(1000, 400, 2000, 1200), track_id=tid) for i in range(n)]


def pass_sets(events):
    return sorted(tuple(sorted((p.camera, p.track_id) for p in e.passes)) for e in events)


def two_trucks():
    return {"561": track("561", 1, 10, 20) + track("561", 2, 100, 110),
            "558": track("558", 7, 15, 30) + track("558", 8, 105, 120)}


def test_replay_matches_batch():
    tracks = two_trucks()
    batch = extract_events(tracks, CONFIGS, RECS).events
    live = LiveEventAssembler(CONFIGS, RECS).replay(tracks)
    assert pass_sets(e for _, e in live) == pass_sets(batch) == [(("558", 7), ("561", 1)),
                                                                (("558", 8), ("561", 2))]


def test_event_is_final_about_a_minute_after_the_truck_leaves():
    live = LiveEventAssembler(CONFIGS, RECS)
    finalised = live.replay(two_trucks())
    first_time, first = finalised[0]
    assert pass_sets([first]) == [(("558", 7), ("561", 1))]
    lag = (first_time - first.end).total_seconds()
    assert live.final_after_end_s < lag < live.final_after_end_s + 10


def test_provisional_until_final_and_reported_once():
    live = LiveEventAssembler(CONFIGS, RECS)
    for cam, dets in {"561": track("561", 1, 0, 10), "558": track("558", 7, 5, 20)}.items():
        for d in dets:
            live.add_frame(cam, d.t, [d])
    for cam in CONFIGS:
        live.add_frame(cam, 30.0, [])                  # tracks ended, event not yet final
    up = live.update()
    assert len(up.provisional) == 1 and not up.finalized
    for cam in CONFIGS:
        live.add_frame(cam, 200.0, [])
    up = live.update()
    assert len(up.finalized) == 1 and not up.provisional
    assert not live.update().finalized                 # never reported twice


def test_track_in_progress_holds_the_event():
    live = LiveEventAssembler(CONFIGS, RECS)
    for d in track("561", 1, 0, 10):
        live.add_frame("561", d.t, [d])
    for d in track("558", 7, 5, 300):                 # truck stands under 558 for minutes
        live.add_frame("558", d.t, [d])
    live.add_frame("561", 300.0, [])
    assert not live.update().finalized


def test_slowest_camera_sets_now():
    live = LiveEventAssembler(CONFIGS, RECS)
    live.add_frame("561", 100.0, [])
    assert live.now() is None                          # 558 has not reported yet
    live.add_frame("558", 40.0, [])
    assert live.now() == datetime(2026, 11, 13, 9, 0, 40)


def test_event_id_is_stable():
    e = extract_events(two_trucks(), CONFIGS, RECS).events[0]
    assert event_id(e) == "20261113-090010_561_1"


FOOTAGE_TRACKS = root_dir / "results" / "events" / "tracks.csv"


@pytest.mark.skipif(not FOOTAGE_TRACKS.exists() or not (root_dir / "videos").exists(),
                    reason="12 Aug tracks/footage not present (gitignored)")
def test_replay_of_the_12_aug_hour_gives_the_batch_events():
    configs = load_camera_configs(root_dir / "config" / "cameras.yaml")
    recs = {c: parse_recording_name(p.name) for c, p in find_recordings(root_dir / "videos").items()}
    tracks = load_tracks_csv(FOOTAGE_TRACKS)
    batch = extract_events(tracks, configs, recs).events
    live = LiveEventAssembler(configs, recs).replay(tracks)
    assert len(batch) == 16
    assert pass_sets(e for _, e in live) == pass_sets(batch)
    assert all(t is not None for t, _ in live)        # all final before the recording ended
