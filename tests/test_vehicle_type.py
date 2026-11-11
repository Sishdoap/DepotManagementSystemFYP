"""Tests for the vehicle type classifier wrapper and per-truck fusion."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.vehicle_type import (
    CLASSES, UNKNOWN, FrameType, UltralyticsVehicleTypeClassifier, fuse_vehicle_type,
    label_to_class, load_frame_types, sha256_of, single_frame_type,
)


def probs(**kw):
    p = {c: 0.0 for c in CLASSES}
    p.update(kw)
    return p


def ft(camera, frame, box_px=1000, **kw):
    return FrameType(camera, frame, probs(**kw), box_px)


class TestLabels:
    def test_sheet_words_map_to_classes(self):
        assert label_to_class("container_truck") == "container_carrier"
        assert label_to_class("box_lorry") == "box_truck"
        assert label_to_class(" Light_Vehicle ") == "light_vehicle"

    def test_blank_or_unknown_label(self):
        assert label_to_class("") is None
        assert label_to_class("forklift") is None


class TestFuse:
    def test_mean_over_frames_beats_a_bad_frame(self):
        frames = [ft("561", 0, container_carrier=0.9, box_truck=0.1),
                  ft("557", 1, container_carrier=0.8, other=0.2),
                  ft("558", 2, box_truck=0.7, container_carrier=0.3)]   # one bad frame
        out = fuse_vehicle_type(frames)
        assert out.vehicle_type == "container_carrier"
        assert out.conf == pytest.approx(2.0 / 3)
        assert (out.frames, out.cameras) == (3, ("557", "558", "561"))

    def test_low_mean_is_unknown_but_keeps_top(self):
        frames = [ft("561", 0, tanker=0.5, container_carrier=0.5),
                  ft("561", 1, tanker=0.6, other=0.4)]
        out = fuse_vehicle_type(frames, min_conf=0.6)
        assert out.vehicle_type == UNKNOWN and out.top == "tanker"
        assert out.conf == pytest.approx(0.55)

    def test_per_camera_stops_one_camera_outvoting_the_rest(self):
        frames = ([ft("560", i, light_vehicle=0.7, container_carrier=0.3) for i in range(10)]
                  + [ft("557", 100, container_carrier=0.9, other=0.1),
                     ft("561", 101, container_carrier=0.8, other=0.2)])
        assert fuse_vehicle_type(frames, min_conf=0).top == "light_vehicle"
        out = fuse_vehicle_type(frames, per_camera=True)
        assert out.vehicle_type == "container_carrier"
        assert out.conf == pytest.approx((0.3 + 0.9 + 0.8) / 3)
        assert out.frames == 12

    def test_camera_filter(self):
        frames = [ft("561", 0, light_vehicle=1.0), ft("557", 1, box_truck=1.0)]
        assert fuse_vehicle_type(frames, cameras=["557"]).vehicle_type == "box_truck"

    def test_no_frames(self):
        out = fuse_vehicle_type([])
        assert out.vehicle_type == UNKNOWN and out.frames == 0

    def test_single_frame_uses_largest_box(self):
        frames = [ft("561", 0, box_px=800, tanker=1.0), ft("561", 1, box_px=1500, box_truck=0.9, other=0.1),
                  ft("557", 2, box_px=3000, other=1.0)]
        assert single_frame_type(frames, "561").vehicle_type == "box_truck"
        assert single_frame_type(frames).vehicle_type == "other"
        assert single_frame_type(frames, "560").vehicle_type == UNKNOWN


def test_load_frame_types(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text("event,camera,track_id,frame,box_px," + ",".join(CLASSES) + "\n"
                    "3,561,7,120,900,0.1,0,0.8,0.05,0.05\n")
    out = load_frame_types(path)
    assert list(out) == [3]
    f = out[3][0]
    assert (f.camera, f.frame, f.box_px, f.top) == ("561", 120, 900.0, "box_truck")


class TestClassifier:
    def write_card(self, tmp_path, digest, classes=CLASSES):
        weights = tmp_path / "w.pt"
        if not weights.exists():
            weights.write_bytes(b"not really weights")
        card = tmp_path / "card.json"
        card.write_text(json.dumps({"classes": list(classes), "weights_sha256": digest}))
        return weights, card

    def test_sha_mismatch_refused(self, tmp_path):
        weights, card = self.write_card(tmp_path, "0" * 64)
        with pytest.raises(RuntimeError, match="does not match"):
            UltralyticsVehicleTypeClassifier(weights, card)

    def test_class_order_mismatch_refused(self, tmp_path):
        weights, card = self.write_card(tmp_path, "", classes=sorted(CLASSES))
        with pytest.raises(RuntimeError, match="classes"):
            UltralyticsVehicleTypeClassifier(weights, card)

    def test_sha256_of(self, tmp_path):
        p = tmp_path / "x"
        p.write_bytes(b"abc")
        assert sha256_of(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_probabilities_mapped_by_name_and_bgr_fed(self):
        # model.names is alphabetical; raw output index i belongs to that name.
        alphabetical = sorted(CLASSES)
        seen = {}

        class FakeModel:
            names = dict(enumerate(alphabetical))

            def predict(self, imgs, **kw):
                seen["first_pixel"] = imgs[0][0, 0].tolist()
                raw = np.array([{"tanker": 3.0, "other": 1.0}.get(n, 0.0) for n in alphabetical])
                data = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: raw))
                return [SimpleNamespace(probs=SimpleNamespace(data=data)) for _ in imgs]

        clf = UltralyticsVehicleTypeClassifier.__new__(UltralyticsVehicleTypeClassifier)
        clf.model, clf.device = FakeModel(), None
        inv = {n: i for i, n in clf.model.names.items()}
        clf.order = [inv[c] for c in CLASSES]

        crop = np.zeros((4, 4, 3), dtype=np.uint8)
        crop[0, 0] = (255, 0, 0)                                   # red in RGB
        p = clf.predict(crop)
        assert seen["first_pixel"] == [0, 0, 255]                  # handed over as BGR
        assert list(p) == list(CLASSES)
        assert p["tanker"] == pytest.approx(0.75) and p["other"] == pytest.approx(0.25)

    def test_bad_crop_rejected(self):
        clf = UltralyticsVehicleTypeClassifier.__new__(UltralyticsVehicleTypeClassifier)
        with pytest.raises(ValueError):
            clf.predict(np.zeros((0, 0, 3), dtype=np.uint8))
