"""Detect and track trucks on several live cameras with one detector (live mode).

scripts/track_trucks.py runs one process per video, each with its own
YOLO model calling `model.track(frame, persist=True)`. Live, five such
processes would each hold a CUDA context next to PaddleOCR and the user's
snapshot recorder on one 16 GB GPU — the memory squeeze that froze the PC
on 2026-11-13. So here one `Detector` serves every camera (frames are
batched across cameras) and each camera keeps its own ByteTrack state.
Ultralytics' tracking callback does the same per frame: run the detector,
then `tracker.update(boxes)`, and report the tracker's boxes with their
track ids — so the tracks match the batch tracker's (same model, imgsz,
confidence threshold, classes and tracker config).

Per tracked detection the frame processor also
- classifies the vehicle type (src.vehicle_type, RGB crop = box + 10%)
  on cameras that are not ignored, when the box's short side is at least
  `min_type_px` — the same frames scripts/classify_events.py uses;
- saves the crop for OCR (`CropSpool`) when the box is big enough for
  codes or plates on that camera's role (src.live_ocr).

`Detector` and the tracker factory are seams: tests use fakes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .footage import Detection, crop_box
from .live_ocr import CODE_ROLES, PLATE_ROLES, SpooledCrop
from .vehicle_type import FrameType

# COCO classes a container truck is detected as (as scripts/track_trucks.py).
VEHICLE_CLASSES = {5: "bus", 6: "train", 7: "truck"}


class Detector(ABC):
    """Batched object detection; returns per image an object the tracker accepts."""

    @abstractmethod
    def detect(self, images_bgr: Sequence[np.ndarray]) -> list:
        ...


class UltralyticsDetector(Detector):
    """YOLO with the batch tracker's settings; returns numpy `Boxes` per image."""

    def __init__(self, model_path="models/yolov8n.pt", imgsz: int = 1280, conf: float = 0.1,
                 classes=tuple(VEHICLE_CLASSES), device=0):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.kw = dict(imgsz=imgsz, conf=conf, classes=list(classes), device=device, verbose=False)

    def detect(self, images_bgr):
        return [r.boxes.cpu().numpy() for r in self.model.predict(list(images_bgr), **self.kw)]


def make_bytetrack():
    """A ByteTrack tracker configured like `model.track(tracker="bytetrack.yaml")`."""
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.utils.checks import check_yaml
    return BYTETracker(args=IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml"))))


class MultiCameraTracker:
    """One detector, one tracker per camera."""

    def __init__(self, detector: Detector, tracker_factory: Callable = make_bytetrack,
                 labels: Mapping[int, str] = VEHICLE_CLASSES):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.labels = labels
        self.trackers: dict[str, object] = {}

    def step(self, frames) -> dict[str, list[Detection]]:
        """Track one sampled frame per camera (`frames`: SampledFrame-like objects)."""
        frames = list(frames)
        if not frames:
            return {}
        results = self.detector.detect([f.image for f in frames])
        out: dict[str, list[Detection]] = {}
        for f, det in zip(frames, results):
            tracker = self.trackers.get(f.camera)
            if tracker is None:
                tracker = self.trackers[f.camera] = self.tracker_factory()
            tracks = tracker.update(det, f.image)
            dets = []
            for row in np.asarray(tracks).reshape(-1, 8):
                x1, y1, x2, y2, tid, score, cls, _ = row
                dets.append(Detection(t=f.t, frame=f.frame, label=self.labels.get(int(cls), str(int(cls))),
                                      conf=float(score), box=(float(x1), float(y1), float(x2), float(y2)),
                                      track_id=int(tid)))
            out[f.camera] = dets
        return out


class CropSpool:
    """Saves truck crops for later OCR; one JPEG per (camera, track, frame)."""

    def __init__(self, root, roles: Mapping[str, str], *, min_box_px: float = 600.0,
                 quality: int = 95):
        self.root = Path(root)
        self.roles = roles
        self.min_box_px = min_box_px
        self.quality = quality

    def wanted(self, camera: str, d: Detection) -> bool:
        role = self.roles.get(camera, "ignore")
        return (role in CODE_ROLES or role in PLATE_ROLES) and d.longest_side >= self.min_box_px

    def save(self, camera: str, d: Detection, image_bgr: np.ndarray) -> Optional[SpooledCrop]:
        if not self.wanted(camera, d):
            return None
        import cv2
        crop, origin = crop_box(image_bgr, d.box)
        if crop.size == 0:
            return None
        path = self.root / camera / f"{d.track_id}_{d.frame}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return SpooledCrop(camera, d.track_id, d.frame, d.t, d.box, origin, str(path))

    def remove(self, crops) -> None:
        for c in crops:
            try:
                os.remove(c.path)
            except FileNotFoundError:
                pass


def type_crops(image_bgr: np.ndarray, dets: Sequence[Detection], min_type_px: float = 64.0
               ) -> tuple[list[Detection], list[np.ndarray]]:
    """Detections big enough to classify, and their RGB crops (box + 10%)."""
    keep, crops = [], []
    for d in dets:
        x1, y1, x2, y2 = d.box
        if min(x2 - x1, y2 - y1) < min_type_px:
            continue
        crop, _ = crop_box(image_bgr, d.box)
        if crop.size:
            keep.append(d)
            crops.append(np.ascontiguousarray(crop[:, :, ::-1]))
    return keep, crops


def frame_types(classifier, camera: str, dets: Sequence[Detection],
                crops_rgb: Sequence[np.ndarray]) -> dict[tuple[int, int], FrameType]:
    """{(track id, frame): FrameType} for the classified detections."""
    if not dets:
        return {}
    probs = classifier.predict_batch(list(crops_rgb))
    return {(d.track_id, d.frame): FrameType(camera, d.frame, p, d.longest_side)
            for d, p in zip(dets, probs)}
