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