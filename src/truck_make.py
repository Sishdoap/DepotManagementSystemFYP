"""Truck make per truck (FR3, stretch): cab-front crop classifier + averaging.

The classifier is trained in the separate repository
(truck_vision_pipeline_claude, `scripts/`, report
`reports/make_evaluation.md`). Hand-off: `models/truck_make.pt`
(Ultralytics 8.4.192, yolo11s-cls, imgsz 320) and
`models/truck_make_card.json` (classes, calibration, unknown threshold,
sha256 of the weights), both gitignored like the other weights.

Classes (the card's order): scania, volvo, hino, other_make,
make_not_visible. Makes with fewer than 50 training visits were merged into
other_make; make_not_visible means the make cannot be told from the crop
(no cab front, or too dark or blurred at night).

Design decisions, from the card and the other repo's predict.py:

- **Camera 561 only** (the card's "camera 161", head-on at the barrier):
  the model was never trained or tested on other views, so other cameras'
  crops are not classified (`MAKE_CAMERAS`).
- Input is the detector box padded by 10% per side, RGB — the same crop as
  src.footage.crop_box, so it matches the training crops. The card's
  `letterbox` flag says whether to pad to a grey square first (false for
  the shipped model: Ultralytics' own short-side resize + centre crop).
- Probabilities are mapped by class name (model.names is alphabetical),
  then **temperature-calibrated** with the card's T: softmax(log(p) / T).
- Decide per truck: average the calibrated probabilities over the truck's
  561 frames; "unknown" when the top mean is below the card's
  `unknown_threshold` (0.5, chosen by the user, not fitted).
- Only for vehicle types the card lists (`applies_to_types`: container
  carrier, tanker, box truck) — and for "unknown" type, since most unknowns
  on the footage are container trucks. A light vehicle or "other" gets no
  make (None), not a guess.
- The weights are checked against the card's sha256 on load.

**Scoring:** the model was trained on 12–15 and 17 Nov and validated and
calibrated on 16 Nov — this repo's test days. Make is therefore scored on
the other repo's locked test days (18–19 Nov, camera-161 snapshots), not
on 16–17 Nov (decision 2026-11-20). On 16–17 Nov the pipeline still
predicts make, but those predictions are not a result.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .vehicle_type import UNKNOWN, sha256_of

MAKE_CAMERAS = ("561",)
NOT_VISIBLE = "make_not_visible"
DEFAULT_APPLIES_TO = ("container_carrier", "tanker", "box_truck")


class TruckMakeClassifier(ABC):
    """Calibrated make probabilities for cab-front crops."""

    classes: tuple[str, ...]
    unknown_threshold: float = 0.5
    applies_to_types: tuple[str, ...] = DEFAULT_APPLIES_TO

    @abstractmethod
    def predict_batch(self, crops_rgb: Sequence[np.ndarray]) -> list[dict[str, float]]:
        """{make: probability} per RGB uint8 crop (H, W, 3), each summing to 1."""


def calibrate(p: np.ndarray, temperature: float) -> np.ndarray:
    """softmax(log(p) / T): T > 1 softens an overconfident model."""
    z = np.log(np.clip(p / p.sum(), 1e-12, 1.0)) / float(temperature)
    e = np.exp(z - z.max())
    return e / e.sum()


def letterbox(img: np.ndarray, fill: int = 114) -> np.ndarray:
    """Pad to a centred grey square (training preprocessing when the card says letterbox)."""
    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), fill, dtype=img.dtype)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return sq


class UltralyticsTruckMakeClassifier(TruckMakeClassifier):
    """The other repo's make model, called the way its predict_make does.

    ultralytics is imported here, not at module level, so tests and the
    rest of the pipeline do not pay for it.
    """

    def __init__(self, weights="models/truck_make.pt", card="models/truck_make_card.json",
                 device: Optional[str] = None):
        card = json.loads(Path(card).read_text())
        digest = sha256_of(weights)
        if digest != card["weights_sha256"]:
            raise RuntimeError(f"{weights} sha256 {digest[:8]}… does not match the make card "
                               f"({card['weights_sha256'][:8]}…)")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.classes = tuple(card["classes"])
        inv = {name: idx for idx, name in self.model.names.items()}
        missing = set(self.classes) - set(inv)
        if missing:
            raise RuntimeError(f"{weights} lacks classes {missing}")
        self.order = [inv[c] for c in self.classes]
        self.temperature = float(card["calibration"]["temperature"])
        self.unknown_threshold = float(card["unknown_threshold"])
        self.applies_to_types = tuple(card.get("applies_to_types", DEFAULT_APPLIES_TO))
        self.letterbox = bool(card.get("letterbox", False))
        self.card = card

    def predict_batch(self, crops_rgb: Sequence[np.ndarray]) -> list[dict[str, float]]:
        imgs = []
        for c in crops_rgb:
            if c.ndim != 3 or c.shape[2] != 3 or c.size == 0:
                raise ValueError(f"expected an RGB image (H, W, 3), got shape {c.shape}")
            img = letterbox(c) if self.letterbox else c
            imgs.append(np.ascontiguousarray(img[:, :, ::-1]))     # Ultralytics treats numpy as BGR
        out = []
        for r in self.model.predict(imgs, verbose=False, device=self.device):
            p = r.probs.data.cpu().numpy()[self.order].astype(float)
            out.append(dict(zip(self.classes, calibrate(p, self.temperature).tolist())))
        return out


@dataclass(frozen=True)
class FrameMake:
    """Calibrated make probabilities of one frame of a truck."""
    camera: str
    frame: int
    probs: dict


@dataclass
class FusedMake:
    make: Optional[str]          # a class, UNKNOWN, or None when make does not apply
    conf: float = 0.0            # mean probability of the top class
    top: Optional[str] = None    # top class even when below the threshold
    frames: int = 0
    mean: dict = field(default_factory=dict)


def fuse_make(frames: Iterable[FrameMake], vehicle_type: str, *, threshold: float = 0.5,
              applies_to: Iterable[str] = DEFAULT_APPLIES_TO,
              cameras: Iterable[str] = MAKE_CAMERAS) -> FusedMake:
    """Average a truck's make probabilities; UNKNOWN below `threshold`.

    None when the vehicle type has no make in the model's sense (light
    vehicle, other); a truck of applicable or unknown type that 561 never
    classified is UNKNOWN.
    """
    if vehicle_type != UNKNOWN and vehicle_type not in set(applies_to):
        return FusedMake(None)
    keep = set(cameras)
    frames = [f for f in frames if f.camera in keep]
    if not frames:
        return FusedMake(UNKNOWN)
    classes = list(frames[0].probs)
    mean = {c: float(np.mean([f.probs.get(c, 0.0) for f in frames])) for c in classes}
    top = max(classes, key=mean.get)
    return FusedMake(top if mean[top] >= threshold else UNKNOWN, mean[top], top, len(frames), mean)
