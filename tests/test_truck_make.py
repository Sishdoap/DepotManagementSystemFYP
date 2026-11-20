"""Tests for truck make per truck (src/truck_make.py) and its frames (src/live_tracking.py)."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pytest

from src.footage import Detection
from src.live_tracking import frame_makes
from src.truck_make import (FrameMake, TruckMakeClassifier, calibrate, fuse_make, letterbox)
from src.vehicle_type import UNKNOWN

CLASSES = ("scania", "volvo", "hino", "other_make", "make_not_visible")


def probs(**kw):
    p = {c: 0.0 for c in CLASSES}
    p.update(kw)
    return p


class FakeMake(TruckMakeClassifier):
    classes = CLASSES

    def __init__(self):
        self.calls = 0

    def predict_batch(self, crops_rgb):
        self.calls += 1
        return [probs(scania=0.9, volvo=0.1) for _ in crops_rgb]


def test_calibration_softens_and_keeps_order():
    p = np.array([0.9, 0.05, 0.05])
    q = calibrate(p, 2.0)
    assert q.sum() == pytest.approx(1.0)
    assert q.argmax() == 0 and q[0] < 0.9
    assert calibrate(p, 1.0) == pytest.approx(p)


def test_letterbox_pads_to_centred_square():
    img = np.zeros((10, 30, 3), np.uint8)
    sq = letterbox(img)
    assert sq.shape == (30, 30, 3)
    assert (sq[:10] == 114).all() and (sq[10:20] == 0).all() and (sq[20:] == 114).all()


def test_fuse_averages_561_frames_with_threshold():
    frames = [FrameMake("561", 0, probs(scania=0.8, volvo=0.2)),
              FrameMake("561", 5, probs(scania=0.4, volvo=0.6)),
              FrameMake("557", 5, probs(volvo=1.0))]                     # other cameras ignored
    got = fuse_make(frames, "container_carrier")
    assert got.make == "scania" and got.conf == pytest.approx(0.6) and got.frames == 2
    low = fuse_make(frames, "container_carrier", threshold=0.7)
    assert low.make == UNKNOWN and low.top == "scania"


def test_make_only_for_applicable_or_unknown_types():
    frames = [FrameMake("561", 0, probs(hino=1.0))]
    assert fuse_make(frames, "light_vehicle").make is None
    assert fuse_make(frames, "other").make is None
    assert fuse_make(frames, UNKNOWN).make == "hino"
    assert fuse_make(frames, "tanker").make == "hino"
    assert fuse_make([], "container_carrier").make == UNKNOWN               # no 561 frames


def test_frame_makes_only_on_camera_561():
    m = FakeMake()
    det = Detection(1.0, 20, "truck", 0.9, (0, 0, 400, 400), 7)
    crop = np.zeros((8, 8, 3), np.uint8)
    assert frame_makes(m, "557", [det], [crop]) == {} and m.calls == 0
    got = frame_makes(m, "561", [det], [crop])
    assert list(got) == [(7, 20)] and got[(7, 20)].probs["scania"] == 0.9
