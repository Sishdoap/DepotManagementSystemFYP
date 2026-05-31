"""Tests for the image-source layer."""

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.image_source import (
    RealImageSource,
    SourcedImage,
    SyntheticImageSource,
)
from src.synthetic_images import ContainerImageGenerator


@pytest.fixture
def real_image_dir(tmp_path):
    """Create a fake images/ directory with labeled and unlabeled files."""
    d = tmp_path / "images"
    d.mkdir()
    # Create simple 50x50 RGB images with various filename patterns.
    img = Image.new("RGB", (50, 50), (200, 100, 50))
    img.save(d / "0_TEMU6472145_1.jpg")
    img.save(d / "1_CSQU3054383_1.jpg")
    img.save(d / "2_MSCU7894561_2.jpg")
    img.save(d / "unlabeled.png")
    return d


class TestRealImageSource:
    def test_finds_all_images(self, real_image_dir):
        src = RealImageSource(real_image_dir, rng=random.Random(0))
        assert src.n_images == 4

    def test_returns_sourced_image(self, real_image_dir):
        src = RealImageSource(real_image_dir, rng=random.Random(0))
        result = src.next_image()
        assert isinstance(result, SourcedImage)
        assert isinstance(result.image, Image.Image)
        assert result.image.mode == "RGB"

    def test_extracts_ground_truth_from_filename(self, real_image_dir):
        src = RealImageSource(real_image_dir, rng=random.Random(0))
        seen = {}
        for _ in range(4):
            r = src.next_image()
            seen[r.source_id] = r.ground_truth_code
        assert seen["0_TEMU6472145_1.jpg"] == "TEMU6472145"
        assert seen["1_CSQU3054383_1.jpg"] == "CSQU3054383"
        assert seen["2_MSCU7894561_2.jpg"] == "MSCU7894561"
        assert seen["unlabeled.png"] is None

    def test_no_repeats_within_pass(self, real_image_dir):
        src = RealImageSource(real_image_dir, rng=random.Random(0))
        ids = [src.next_image().source_id for _ in range(4)]
        assert len(set(ids)) == 4  # all unique within one pass

    def test_reshuffles_after_pass(self, real_image_dir):
        src = RealImageSource(real_image_dir, rng=random.Random(0))
        # Exhaust then take one more — must not raise.
        for _ in range(5):
            r = src.next_image()
            assert r is not None

    def test_shuffle_order_reproducible(self, real_image_dir):
        s1 = RealImageSource(real_image_dir, rng=random.Random(42))
        s2 = RealImageSource(real_image_dir, rng=random.Random(42))
        ids1 = [s1.next_image().source_id for _ in range(4)]
        ids2 = [s2.next_image().source_id for _ in range(4)]
        assert ids1 == ids2

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RealImageSource(tmp_path / "does_not_exist")

    def test_empty_dir_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError):
            RealImageSource(d)


class TestSyntheticImageSource:
    def test_wraps_generator(self):
        gen = ContainerImageGenerator(rng=random.Random(0))
        src = SyntheticImageSource(gen)
        r = src.next_image()
        assert isinstance(r.image, Image.Image)
        assert r.ground_truth_code is not None    # always known for synthetic
        assert r.source_id.startswith("synthetic-")