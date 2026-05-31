"""Image source abstraction for the simulator.

Two implementations:
    SyntheticImageSource — wraps the existing ContainerImageGenerator.
    RealImageSource      — samples from a directory of labeled container photos.

Both return PIL.Image.Image; downstream code (OCR adapter) handles the
PIL <-> ndarray conversion as needed.
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .synthetic_images import (
    ContainerImageGenerator,
    GenerationConfig,
)


# Filenames like "0_TEMU6472145_1.jpg" or "TEMU6472145.png".
# Captures the 11-char ISO 6346 code if present.
_FILENAME_CODE_RE = re.compile(
    r"^(?:\d+_)?([A-Z]{4}\d{7})(?:_\d+)?\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourcedImage:
    """One image pulled from an ImageSource.

    Attributes:
        image: the image as PIL. Downstream converts to ndarray if needed.
        ground_truth_code: the labeled ISO 6346 code if known, else None.
        source_id: human-readable identifier (filename, synthetic id, etc.)
            — written to the database for traceability.
    """
    image: Image.Image
    ground_truth_code: Optional[str]
    source_id: str


class ImageSource(ABC):
    """Anything that produces container images for the simulator."""

    @abstractmethod
    def next_image(self) -> SourcedImage:
        """Return the next image. Must be safe to call repeatedly."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging."""


# --- Synthetic (wraps the existing generator) ---

class SyntheticImageSource(ImageSource):
    """Wraps ContainerImageGenerator behind the ImageSource interface."""

    def __init__(
        self,
        generator: ContainerImageGenerator,
        config: Optional[GenerationConfig] = None,
    ):
        self._generator = generator
        self._config = config or GenerationConfig()
        self._counter = 0

    @property
    def name(self) -> str:
        return "synthetic"

    def next_image(self) -> SourcedImage:
        result = self._generator.generate(self._config)
        self._counter += 1
        # GeneratedImage.image is ndarray; convert to PIL for the interface.
        pil = Image.fromarray(result.image)
        return SourcedImage(
            image=pil,
            ground_truth_code=result.true_code,
            source_id=f"synthetic-{self._counter:06d}",
        )


# --- Real images from a directory ---

class RealImageSource(ImageSource):
    """Samples from a directory of real container photos with shuffle-and-iterate.

    Strategy: load file list once at construction, shuffle deterministically
    (via the passed RNG), iterate to end, reshuffle, repeat. This guarantees
    no repeats within a single pass through the dataset — better for demos
    than i.i.d. random sampling.
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        image_dir: str | Path = "images",
        rng: Optional[random.Random] = None,
    ):
        self._dir = Path(image_dir)
        self._rng = rng or random.Random()
        if not self._dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self._dir.resolve()}"
            )

        # Discover all images once.
        self._files: list[Path] = sorted(
            p for p in self._dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
        )
        if not self._files:
            raise ValueError(
                f"No images found in {self._dir.resolve()} "
                f"(looking for {sorted(self.SUPPORTED_EXTS)})"
            )

        # Internal shuffle state.
        self._order: list[int] = []
        self._cursor = 0
        self._reshuffle()

    @property
    def name(self) -> str:
        return f"real:{self._dir.name}"

    @property
    def n_images(self) -> int:
        return len(self._files)

    def next_image(self) -> SourcedImage:
        if self._cursor >= len(self._order):
            self._reshuffle()
        idx = self._order[self._cursor]
        self._cursor += 1

        path = self._files[idx]
        pil = Image.open(path).convert("RGB")
        return SourcedImage(
            image=pil,
            ground_truth_code=self._parse_code_from_filename(path.name),
            source_id=path.name,
        )

    def _reshuffle(self) -> None:
        self._order = list(range(len(self._files)))
        self._rng.shuffle(self._order)
        self._cursor = 0

    @staticmethod
    def _parse_code_from_filename(filename: str) -> Optional[str]:
        """Extract ISO 6346 code from `0_TEMU6472145_1.jpg` style filenames.

        Returns None if no recognizable code is present.
        """
        m = _FILENAME_CODE_RE.match(filename)
        return m.group(1).upper() if m else None