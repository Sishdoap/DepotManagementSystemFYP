"""Vehicle type per truck (FR3): crop classifier + per-truck averaging.

The classifier is trained in a separate repository
(truck_vision_pipeline_claude) on public Roboflow datasets plus the
collaborator's EOR gate images — never on this project's depot footage,
so the footage stays a clean held-out test. The hand-off is
models/vehicle_type.pt (Ultralytics 8.4.192, yolo11s-cls) and
models/model_card.json (classes, metrics, sha256 of the weights); both are
gitignored like the other weights.

Classes (CLASSES, the card's order):
    container_carrier  prime mover + container chassis, laden or empty
    tanker             tank trailer / tank truck
    box_truck          rigid lorry with a box body
    light_vehicle      cars, SUVs, pickups, vans
    other              buses, cranes, flatbeds, anything else
"other" is a fifth class added in the classifier repo (the brief planned
four): without it, everything that is not one of the four is forced into
one of them.

Design decisions, mostly from the hand-off notes:

- Input is the detector box padded by 10% per side, RGB. That is the same
  crop as src.footage.crop_box (also used for OCR), so the crops match the
  training crops. OpenCV frames are BGR: callers convert before
  `predict`. A wrong colour order does not crash; it just quietly makes
  results worse.
- Ultralytics stores class probabilities in alphabetical order
  (model.names). Probabilities are always mapped by class name, never by
  position.
- Decide per truck, not per frame (`fuse_vehicle_type`): the class
  probabilities of every frame of the truck's event, on every camera, are
  averaged, and the top class is kept only if its mean is at least
  `min_conf` (0.6); otherwise the truck is "unknown" and goes to a manual
  check rather than a forced class. Averaging smooths out the bad frames
  (cab only, glare, half out of frame).
- The weights file is checked against the card's sha256 on load, so a
  stale copy cannot be evaluated by mistake.

The label sheet (scripts/make_label_sheet.py) uses its own words for the
types; LABEL_TO_CLASS maps them.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

CLASSES = ("container_carrier", "tanker", "box_truck", "light_vehicle", "other")
UNKNOWN = "unknown"
LABEL_TO_CLASS = {
    "container_truck": "container_carrier",
    "container_carrier": "container_carrier",
    "tanker": "tanker",
    "box_lorry": "box_truck",
    "box_truck": "box_truck",
    "light_vehicle": "light_vehicle",
    "other": "other",
}


def label_to_class(label: str) -> Optional[str]:
    """Label sheet vehicle_type -> classifier class, None if blank/unknown."""
    return LABEL_TO_CLASS.get((label or "").strip().lower())


class VehicleTypeClassifier(ABC):
    """Class probabilities for one vehicle crop."""

    @abstractmethod
    def predict(self, crop_rgb: np.ndarray) -> dict[str, float]:
        """{class: probability} over CLASSES, summing to 1. `crop_rgb` is RGB uint8 (H, W, 3)."""


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class UltralyticsVehicleTypeClassifier(VehicleTypeClassifier):
    """The classifier repo's model, called the way its predict.py does.

    ultralytics is imported here, not at module level, so tests and the
    rest of the pipeline do not pay for it.
    """

    def __init__(self, weights="models/vehicle_type.pt", card="models/model_card.json",
                 device: Optional[str] = None):
        card = json.loads(Path(card).read_text())
        if tuple(card["classes"]) != CLASSES:
            raise RuntimeError(f"model card classes {card['classes']} != {CLASSES}")
        digest = sha256_of(weights)
        if digest != card["weights_sha256"]:
            raise RuntimeError(f"{weights} sha256 {digest[:8]}… does not match the model card "
                               f"({card['weights_sha256'][:8]}…)")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        inv = {name: idx for idx, name in self.model.names.items()}
        missing = set(CLASSES) - set(inv)
        if missing:
            raise RuntimeError(f"{weights} lacks classes {missing}")
        self.order = [inv[c] for c in CLASSES]
        self.card = card

    def predict(self, crop_rgb: np.ndarray) -> dict[str, float]:
        return self.predict_batch([crop_rgb])[0]

    def predict_batch(self, crops_rgb: list) -> list[dict[str, float]]:
        for c in crops_rgb:
            if c.ndim != 3 or c.shape[2] != 3 or c.size == 0:
                raise ValueError(f"expected an RGB image (H, W, 3), got shape {c.shape}")
        # Ultralytics treats numpy input as BGR.
        bgr = [np.ascontiguousarray(c[:, :, ::-1]) for c in crops_rgb]
        out = []
        for r in self.model.predict(bgr, verbose=False, device=self.device):
            p = r.probs.data.cpu().numpy()[self.order].astype(float)
            out.append(dict(zip(CLASSES, (p / p.sum()).tolist())))
        return out


@dataclass(frozen=True)
class FrameType:
    """Class probabilities of one frame of a truck."""
    camera: str
    frame: int
    probs: dict
    box_px: float = 0.0          # longest side of the detector box

    @property
    def top(self) -> str:
        return max(CLASSES, key=lambda c: self.probs.get(c, 0.0))


@dataclass
class FusedType:
    vehicle_type: str            # a class from CLASSES, or UNKNOWN
    conf: float = 0.0            # mean probability of the top class
    top: Optional[str] = None    # top class even when below min_conf
    frames: int = 0
    cameras: tuple[str, ...] = ()
    mean: dict = field(default_factory=dict)


def fuse_vehicle_type(frames: Iterable[FrameType], *, min_conf: float = 0.6,
                      cameras: Optional[Iterable[str]] = None,
                      per_camera: bool = False) -> FusedType:
    """Average the class probabilities over all frames; UNKNOWN below `min_conf`.

    `cameras` restricts the average to those cameras (for per-camera
    comparisons). By default every frame counts once. With `per_camera`,
    each camera's frames are averaged first and the cameras then count
    equally: the number of frames a camera contributes reflects how long
    the truck stood in its view, not how informative the view is. On the
    footage, a truck waiting under side camera 560 gave ~400 frames of the
    out-of-focus top of its cab, which outvoted every other camera. (That
    variant was chosen after seeing the footage results; see
    scripts/fuse_vehicle_types.py.)
    """
    keep = set(cameras) if cameras is not None else None
    frames = [f for f in frames if keep is None or f.camera in keep]
    if not frames:
        return FusedType(UNKNOWN)
    if per_camera:
        by_cam = defaultdict(list)
        for f in frames:
            by_cam[f.camera].append(f)
        cam_means = [{c: np.mean([f.probs.get(c, 0.0) for f in fs]) for c in CLASSES}
                     for fs in by_cam.values()]
        mean = {c: float(np.mean([m[c] for m in cam_means])) for c in CLASSES}
    else:
        mean = {c: float(np.mean([f.probs.get(c, 0.0) for f in frames])) for c in CLASSES}
    top = max(CLASSES, key=mean.get)
    return FusedType(top if mean[top] >= min_conf else UNKNOWN, mean[top], top, len(frames),
                     tuple(sorted({f.camera for f in frames})), mean)


def single_frame_type(frames: Iterable[FrameType], camera: Optional[str] = None,
                      *, min_conf: float = 0.6) -> FusedType:
    """Baseline: the class of the one frame with the largest box (optionally on one camera)."""
    frames = [f for f in frames if camera is None or f.camera == camera]
    if not frames:
        return FusedType(UNKNOWN)
    best = max(frames, key=lambda f: f.box_px)
    p = best.probs.get(best.top, 0.0)
    return FusedType(best.top if p >= min_conf else UNKNOWN, p, best.top, 1, (best.camera,),
                     dict(best.probs))


def load_frame_types(path) -> dict[int, list[FrameType]]:
    """Read scripts/classify_events.py output: {event: [FrameType, ...]}."""
    import csv

    out: dict[int, list[FrameType]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["event"])].append(FrameType(
                r["camera"], int(r["frame"]), {c: float(r[c]) for c in CLASSES},
                float(r.get("box_px") or 0)))
    return dict(out)
