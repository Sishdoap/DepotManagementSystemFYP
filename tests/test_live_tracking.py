"""Tests for multi-camera tracking, crop spooling and per-frame types (src/live_tracking.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from src.footage import Detection
from src.live_tracking import (CropSpool, Detector, MultiCameraTracker, frame_types, type_crops)


class FakeDetector(Detector):
    def __init__(self):
        self.batches = []

    def detect(self, images_bgr):
        self.batches.append(len(images_bgr))
        return [f"det{i}" for i in range(len(images_bgr))]


class FakeTracker:
    """Returns one track per update, numbered per tracker instance."""
    made = 0

    def __init__(self):
        FakeTracker.made += 1
        self.n = 0

    def update(self, det, img):
        self.n += 1
        return np.array([[10, 20, 710, 320, self.n, 0.8, 7, 0]], dtype=float)


def frame(camera, t=0.0, idx=0):
    return SimpleNamespace(camera=camera, t=t, frame=idx, image=np.zeros((400, 800, 3), np.uint8))


def test_one_batch_and_a_tracker_per_camera():
    FakeTracker.made = 0
    det = FakeDetector()
    mct = MultiCameraTracker(det, FakeTracker)
    out = mct.step([frame("557", 1.0, 20), frame("558", 1.0, 20)])
    out2 = mct.step([frame("557", 1.25, 25)])
    assert det.batches == [2, 1]
    assert FakeTracker.made == 2
    d = out["557"][0]
    assert d == Detection(t=1.0, frame=20, label="truck", conf=0.8, box=(10, 20, 710, 320), track_id=1)
    assert out2["557"][0].track_id == 2          # 557's tracker kept its state
    assert mct.step([]) == {}


def test_crop_spool_by_role_and_size(tmp_path):
    spool = CropSpool(tmp_path, {"557": "side", "561": "cab", "556": "ignore"}, min_box_px=600)
    img = np.full((400, 900, 3), 128, np.uint8)
    big = Detection(t=1.0, frame=4, label="truck", conf=0.9, box=(10, 10, 710, 310), track_id=3)
    small = Detection(t=1.0, frame=4, label="truck", conf=0.9, box=(10, 10, 300, 200), track_id=4)
    c = spool.save("557", big, img)
    assert c is not None and Path(c.path).exists()
    assert c.track_id == 3 and c.frame == 4 and c.origin == (0, 0)
    assert spool.save("557", small, img) is None
    assert spool.save("556", big, img) is None
    assert spool.save("561", big, img) is not None
    spool.remove([c])
    assert not Path(c.path).exists()
    spool.remove([c])                              # already gone: no error


class FakeClassifier:
    def predict_batch(self, crops):
        return [{"container_carrier": 0.9, "other": 0.1} for _ in crops]


def test_type_crops_and_frame_types():
    img = np.zeros((400, 900, 3), np.uint8)
    img[:, :, 0] = 255                             # BGR blue
    dets = [Detection(1.0, 4, "truck", 0.9, (10, 10, 210, 110), 1),
            Detection(1.0, 4, "truck", 0.9, (10, 10, 50, 300), 2)]      # 40 px wide: skipped
    keep, crops = type_crops(img, dets, min_type_px=64)
    assert [d.track_id for d in keep] == [1]
    assert crops[0][0, 0, 2] == 255                # converted to RGB
    ft = frame_types(FakeClassifier(), "557", keep, crops)
    assert list(ft) == [(1, 4)] and ft[(1, 4)].top == "container_carrier"
    assert ft[(1, 4)].box_px == 200
    assert frame_types(FakeClassifier(), "557", [], []) == {}
