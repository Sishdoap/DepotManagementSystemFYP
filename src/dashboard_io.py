"""Helpers for sharing live state between the simulator and the dashboard.

The simulator writes synthetic frames to disk; the dashboard reads them.
This lets the two run as independent processes communicating only through
files and the database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_FRAME_DIR = Path("frames")


def write_gate_frame(
    gate_id: str,
    image: np.ndarray,
    *,
    frame_dir: Path | str = DEFAULT_FRAME_DIR,
    annotation: Optional[str] = None,
    bounding_box: Optional[tuple[int, int, int, int]] = None,
) -> None:
    """Save the latest synthetic image for a gate so the dashboard can show it.

    Writes are atomic: we save to a .tmp file first, then rename. The rename
    is atomic on the same filesystem, so any concurrent reader sees either
    the previous complete frame or the new complete frame — never a partial
    file. Without this, a fast dashboard refresh during a write yields a
    truncated PNG and PIL throws OSError.
    """
    import os
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image)

    # Draw the bounding box first, so the annotation strip overlays it cleanly
    # if there happens to be overlap.
    if bounding_box is not None:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        x, y, w, h = bounding_box
        # Three-pixel-wide green box. Pillow doesn't natively support "width"
        # on rectangle outlines, so we draw concentric rectangles for thickness.
        for offset in range(3):
            draw.rectangle(
                [x - offset, y - offset, x + w + offset, y + h + offset],
                outline=(0, 255, 0),
            )

    if annotation:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, img.height - 25, img.width - 5, img.height - 5],
                       fill=(0, 0, 0, 200))
        draw.text((10, img.height - 22), annotation, fill=(0, 255, 0))

    final_path = frame_dir / f"gate_{gate_id}.png"
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    img.save(tmp_path, format="PNG")
    os.replace(tmp_path, final_path)


def read_gate_frame(
    gate_id: str,
    *,
    frame_dir: Path | str = DEFAULT_FRAME_DIR,
) -> Optional[bytes]:
    """Read the latest frame for a gate. Returns None if no frame exists."""
    path = Path(frame_dir) / f"gate_{gate_id}.png"
    if not path.exists():
        return None
    return path.read_bytes()


def clear_frames(frame_dir: Path | str = DEFAULT_FRAME_DIR) -> None:
    """Remove all gate frame files. Call between runs."""
    frame_dir = Path(frame_dir)
    if not frame_dir.exists():
        return
    for p in frame_dir.glob("gate_*.png"):
        p.unlink()

# --- Depot cameras (FYP2 live mode) ------------------------------------------
# scripts/run_live.py writes these files; scripts/depot_dashboard.py reads them.
# Same rule as above: every write goes to a temp file first, then an atomic
# rename, so the dashboard never reads half a JPEG.

DEFAULT_LIVE_DIR = Path("frames/live")


def _atomic_write(path: Path, data: bytes) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _jpeg(image_bgr: np.ndarray, max_side: int, quality: int = 80) -> bytes:
    import cv2
    h, w = image_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def write_camera_preview(camera: str, image_bgr: np.ndarray, boxes=(), *,
                         live_dir: Path | str = DEFAULT_LIVE_DIR, max_side: int = 1280,
                         label: Optional[str] = None) -> Path:
    """Downscaled frame of one camera with tracked boxes drawn (`boxes`: (x1, y1, x2, y2, text))."""
    import cv2
    img = image_bgr.copy()
    thick = max(2, img.shape[1] // 600)
    for x1, y1, x2, y2, text in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), thick)
        if text:
            cv2.putText(img, str(text), (int(x1), max(30, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        thick * 0.8, (0, 220, 0), thick)
    if label:
        cv2.putText(img, label, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, thick * 0.9,
                    (255, 255, 255), thick)
    path = Path(live_dir) / f"cam_{camera}.jpg"
    _atomic_write(path, _jpeg(img, max_side))
    return path


def read_camera_preview(camera: str, *, live_dir: Path | str = DEFAULT_LIVE_DIR
                        ) -> tuple[Optional[bytes], Optional[float]]:
    """(JPEG bytes, file modification time) of a camera's latest preview, or (None, None)."""
    path = Path(live_dir) / f"cam_{camera}.jpg"
    try:
        return path.read_bytes(), path.stat().st_mtime
    except FileNotFoundError:
        return None, None


def thumbnail_path(pass_uid: str, camera: str, *, live_dir: Path | str = DEFAULT_LIVE_DIR) -> Path:
    return Path(live_dir) / "thumbs" / f"{pass_uid}_{camera}.jpg"


def write_pass_thumbnail(pass_uid: str, camera: str, crop_bgr: np.ndarray, *,
                         live_dir: Path | str = DEFAULT_LIVE_DIR, max_side: int = 640) -> Path:
    """A small picture of a truck pass from one camera (shown on its profile card)."""
    path = thumbnail_path(pass_uid, camera, live_dir=live_dir)
    _atomic_write(path, _jpeg(crop_bgr, max_side))
    return path


def write_status(status: dict, *, live_dir: Path | str = DEFAULT_LIVE_DIR) -> None:
    """The live pipeline's heartbeat (mode, video time, lag, OCR queue) as JSON."""
    import json
    _atomic_write(Path(live_dir) / "status.json", json.dumps(status, default=str).encode())


def read_status(*, live_dir: Path | str = DEFAULT_LIVE_DIR) -> Optional[dict]:
    import json
    try:
        return json.loads((Path(live_dir) / "status.json").read_text())
    except (FileNotFoundError, ValueError):
        return None


def mask_plate(plate: Optional[str]) -> str:
    """Plates are personal data: keep the prefix and last digit, e.g. BQB7404 -> BQB•••4."""
    if not plate:
        return "—"
    prefix = plate[:2] if plate.startswith("T/") else ""
    body = plate[len(prefix):]
    letters = len(body) - len(body.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    keep = body[:letters]
    return prefix + keep + "•" * max(0, len(body) - letters - 1) + body[-1:]
