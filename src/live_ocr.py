"""Read container codes and plates of one finished event from saved crops (live mode).

The batch pipeline reads codes (scripts/read_events.py) and plates
(scripts/read_plates.py) in two passes over the video, seeking to every
tracked frame. Live, the video is a file still being written, so the
tracking stage saves the truck crops the readers need (`SpooledCrop`) and
this module reads them once the event is final (src/live_events.py).

Same crops, same rules as the batch scripts:

- Codes: cameras with role side / codes / backup, boxes whose longest
  side is at least `min_box_px` (600), crop = box + 10% per side, PaddleOCR
  at 0/90/180/270 degrees, `read_container_code` per rotation, best
  rotation per frame, then `MultiContainerFuser`.
- Plates: cameras with role cab / side / codes, same size limit, the
  0-degree OCR output with fragment positions as fractions of the truck
  box, `plates_in_frame`, `fuse_plates` per kind.

The 0-degree OCR call is shared: one PaddleOCR pass gives both the code
fragments and the plate fragments of a frame. That is the only change from
batch, apart from:

- a frame budget (`max_frames_per_camera`): a truck waiting under a side
  camera gives 200+ frames; reading all of them at four rotations takes
  minutes, and the next truck is waiting. Frames are taken evenly spaced
  in time, so arrival (cab plate) and departure (trailer plate) are both
  covered;
- crops are stored as JPEG (quality 95) between tracking and reading.
Whether these change the results is measured by replaying a recorded hour
(scripts/run_live.py --replay) against the batch results.

`TextReader` is the OCR seam: `PaddleTextReader` in production, a fake in
tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .fusion import FrameRead, FusedRead, MultiContainerFuser, best_reading
from .plates import TRAILER, VEHICLE, FusedPlate, Fragment, PlateRead, fuse_plates, plates_in_frame

CODE_ROLES = ("side", "codes", "backup")
PLATE_ROLES = ("cab", "side", "codes")
ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class TextBox:
    """One OCR text line: text, confidence, polygon in image pixels."""
    text: str
    conf: float
    poly: tuple[tuple[float, float], ...]


class TextReader(ABC):
    """Finds and reads text lines in an RGB image."""

    @abstractmethod
    def read(self, image_rgb: np.ndarray) -> list[TextBox]:
        ...


class PaddleTextReader(TextReader):
    """PaddleOCR with the project's settings and the footage memory caps (src/ocr.py)."""

    def __init__(self, use_gpu: bool = True, det_max_side: Optional[int] = 1920,
                 gpu_mem_mb: Optional[int] = 6144):
        import os
        for k in ("FLAGS_use_mkldnn", "FLAGS_enable_pir_in_executor", "FLAGS_enable_pir_api"):
            os.environ.setdefault(k, "0")
        from .ocr import RealOCRAdapter
        self._paddle = RealOCRAdapter._init_paddle(use_gpu, det_max_side=det_max_side,
                                                   gpu_mem_mb=gpu_mem_mb)

    def read(self, image_rgb: np.ndarray) -> list[TextBox]:
        res = self._paddle.predict(np.ascontiguousarray(image_rgb))[0]
        return [TextBox(t, float(c), tuple((float(x), float(y)) for x, y in poly))
                for t, c, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
                if t.strip()]


@dataclass(frozen=True)
class SpooledCrop:
    """A truck crop saved by the tracking stage for later reading."""
    camera: str
    track_id: int
    frame: int
    t: float
    box: tuple[float, float, float, float]    # detection box, full-frame pixels
    origin: tuple[int, int]                   # crop's top-left corner in the frame
    path: str

    @property
    def longest_side(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


def select_frames(crops: Sequence[SpooledCrop], budget: Optional[int]) -> list[SpooledCrop]:
    """At most `budget` crops, evenly spaced in time (all of them if budget is None)."""
    crops = sorted(crops, key=lambda c: (c.t, c.frame))
    if budget is None or len(crops) <= budget:
        return crops
    if budget <= 0:
        return []
    if budget == 1:
        return [crops[len(crops) // 2]]
    idx = sorted({round(i * (len(crops) - 1) / (budget - 1)) for i in range(budget)})
    return [crops[i] for i in idx]


@dataclass
class EventReads:
    """Everything read from one event's crops."""
    code_reads: dict[tuple[str, int], list[FrameRead]] = field(default_factory=dict)
    plate_reads: list[PlateRead] = field(default_factory=list)
    frames_read: int = 0


def read_crop(reader: TextReader, crop_rgb: np.ndarray, crop: SpooledCrop, *,
              codes: bool, plates: bool, rotations: Iterable[int] = ROTATIONS
              ) -> tuple[list[FrameRead], list[PlateRead]]:
    """Code readings (one per rotation) and plate candidates of one crop."""
    from PIL import Image
    from .iso6346_recovery import read_container_code

    frame_reads, plate_reads = [], []
    boxes0 = reader.read(crop_rgb)
    if plates:
        x1, y1, x2, y2 = crop.box
        bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        ox, oy = crop.origin
        frags = []
        for b in boxes0:
            xs = [p[0] + ox for p in b.poly]
            ys = [p[1] + oy for p in b.poly]
            frags.append(Fragment(b.text, b.conf, (min(xs) - x1) / bw, (min(ys) - y1) / bh,
                                  (max(xs) - x1) / bw, (max(ys) - y1) / bh))
        plate_reads = plates_in_frame(frags, camera=crop.camera, frame=crop.frame)
    if codes:
        pil = None
        for rot in rotations:
            if rot == 0:
                boxes = boxes0
            else:
                pil = pil or Image.fromarray(crop_rgb)
                boxes = reader.read(np.asarray(pil.rotate(rot, expand=True)))
            r = read_container_code([(b.text, b.conf) for b in boxes])
            confs = [b.conf for b in boxes]
            frame_reads.append(FrameRead(
                camera=crop.camera, frame=crop.frame, status=r.status, code=r.code or None,
                size_type=r.size_type or None, edits=r.edits,
                conf=round(sum(confs) / len(confs), 3) if confs else 0.0,
                rotation=rot, box_px=crop.longest_side))
    return frame_reads, plate_reads


def read_event(reader: TextReader, crops: Iterable[SpooledCrop], roles: dict[str, str], *,
               min_box_px: float = 600.0, max_frames_per_camera: Optional[int] = 60,
               load=None) -> EventReads:
    """Read every selected crop of one event.

    `roles` maps camera -> role (config/cameras.yaml); `load(path)` returns
    the RGB crop (default: read the JPEG with OpenCV).
    """
    load = load or _load_rgb
    by_cam: dict[str, list[SpooledCrop]] = {}
    for c in crops:
        role = roles.get(c.camera, "ignore")
        if c.longest_side >= min_box_px and (role in CODE_ROLES or role in PLATE_ROLES):
            by_cam.setdefault(c.camera, []).append(c)
    out = EventReads()
    for cam, cs in sorted(by_cam.items()):
        role = roles[cam]
        for c in select_frames(cs, max_frames_per_camera):
            img = load(c.path)
            if img is None:
                continue
            frs, prs = read_crop(reader, img, c, codes=role in CODE_ROLES,
                                 plates=role in PLATE_ROLES)
            if frs:
                out.code_reads[(cam, c.frame)] = frs
            out.plate_reads += prs
            out.frames_read += 1
    return out


def fuse_event(reads: EventReads, fuser=None) -> tuple[FusedRead, FusedPlate, FusedPlate]:
    """Fused code(s), vehicle plate and trailer plate of one event."""
    fuser = fuser or MultiContainerFuser()
    frames = [best_reading(rs) for _, rs in sorted(reads.code_reads.items())]
    return (fuser.fuse(frames), fuse_plates(reads.plate_reads, VEHICLE),
            fuse_plates(reads.plate_reads, TRAILER))


def _load_rgb(path: str):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
