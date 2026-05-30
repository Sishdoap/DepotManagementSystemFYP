"""Tests for the synthetic container image generator."""

import random
import numpy as np
import pytest
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.synthetic_images import (
    CLEAN,
    CONTAINER_COLORS,
    ContainerImageGenerator,
    GenerationConfig,
    GeneratedImage,
    HARSH,
    MODERATE,
    save_image,
)
from src.iso6346 import validate


@pytest.fixture
def gen():
    """Generator with a seeded RNG for reproducibility."""
    return ContainerImageGenerator(rng=random.Random(0))


class TestBasicGeneration:
    def test_returns_generated_image(self, gen):
        result = gen.generate()
        assert isinstance(result, GeneratedImage)

    def test_default_dimensions(self, gen):
        result = gen.generate()
        assert result.image.shape == (100, 400, 3)
        assert result.image.dtype == np.uint8

    def test_custom_dimensions(self, gen):
        config = GenerationConfig(width=600, height=150)
        result = gen.generate(config)
        assert result.image.shape == (150, 600, 3)

    def test_true_code_is_valid(self, gen):
        # Generator must always produce ISO 6346-valid codes.
        for _ in range(50):
            result = gen.generate()
            assert validate(result.true_code).is_valid

    def test_code_override(self, gen):
        result = gen.generate(code_override="CSQU3054383")
        assert result.true_code == "CSQU3054383"

    def test_background_color_recorded(self, gen):
        result = gen.generate()
        assert result.background_color_name in CONTAINER_COLORS


class TestReproducibility:
    def test_same_seed_same_output(self):
        gen1 = ContainerImageGenerator(rng=random.Random(42))
        gen2 = ContainerImageGenerator(rng=random.Random(42))
        r1 = gen1.generate(MODERATE)
        r2 = gen2.generate(MODERATE)
        assert r1.true_code == r2.true_code
        assert r1.background_color_name == r2.background_color_name
        np.testing.assert_array_equal(r1.image, r2.image)

    def test_different_seeds_different_output(self):
        gen1 = ContainerImageGenerator(rng=random.Random(1))
        gen2 = ContainerImageGenerator(rng=random.Random(2))
        r1 = gen1.generate(CLEAN)
        r2 = gen2.generate(CLEAN)
        # At minimum, codes should differ; usually backgrounds too.
        assert r1.true_code != r2.true_code


class TestEffectsApplied:
    def test_clean_vs_noisy_differ(self, gen):
        # Same seed, two configs — noisy output should differ from clean.
        gen2 = ContainerImageGenerator(rng=random.Random(0))
        clean = gen.generate(CLEAN, code_override="CSQU3054383")
        noisy = gen2.generate(
            GenerationConfig(noise_intensity=0.4),
            code_override="CSQU3054383",
        )
        # Same code rendered, but pixel-level noise should make them differ.
        assert not np.array_equal(clean.image, noisy.image)

    def test_rotation_changes_image(self, gen):
        gen2 = ContainerImageGenerator(rng=random.Random(0))
        flat = gen.generate(CLEAN, code_override="CSQU3054383")
        rotated = gen2.generate(
            GenerationConfig(rotation_degrees=10.0),
            code_override="CSQU3054383",
        )
        assert not np.array_equal(flat.image, rotated.image)

    def test_blur_smooths_image(self, gen):
        # Heavy blur should reduce high-frequency content. Measure by std of
        # the Laplacian (a standard sharpness proxy).
        gen2 = ContainerImageGenerator(rng=random.Random(0))
        sharp = gen.generate(CLEAN, code_override="CSQU3054383")
        blurred = gen2.generate(
            GenerationConfig(blur_radius=3.0),
            code_override="CSQU3054383",
        )
        sharp_var = _laplacian_variance(sharp.image)
        blurred_var = _laplacian_variance(blurred.image)
        assert blurred_var < sharp_var

    def test_harsh_preset_doesnt_crash(self, gen):
        # The full feature set must not raise.
        result = gen.generate(HARSH)
        assert result.image.shape == (100, 400, 3)


class TestSaveImage:
    def test_writes_png(self, gen, tmp_path):
        result = gen.generate(CLEAN)
        output = tmp_path / "test_container.png"
        save_image(result, output)
        assert output.exists()
        assert output.stat().st_size > 0


def _laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian — a standard sharpness measure."""
    # Convert to grayscale for sharpness measurement.
    gray = img.mean(axis=2)
    # 5-point Laplacian: center * -4 + neighbors.
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return float(laplacian.var())