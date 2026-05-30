"""Tests for the OCR adapter interface and MockOCRAdapter."""

import random
import numpy as np
import pytest
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ocr import (
    BoundingBox,
    MockOCRAdapter,
    OCRResult,
    RealOCRAdapter,
    STANDARD_CHAR_LIST,
)

from src.iso6346 import validate


@pytest.fixture
def blank_image():
    """A throwaway image — the mock doesn't actually look at pixels."""
    return np.zeros((100, 400, 3), dtype=np.uint8)


class TestMockOCRPerfect:
    """Mock with error_rate=0 should always produce valid codes."""

    def test_returns_ocr_result(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert isinstance(result, OCRResult)

    def test_perfect_ocr_succeeds(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert result.succeeded
        assert result.is_valid
        assert validate(result.recovered_code).is_valid

    def test_perfect_ocr_zero_edits(self, blank_image):
        # No errors injected, raw == recovered.
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert result.raw_string.upper() == result.recovered_code

    def test_distribution_shape(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert result.distributions.shape == (11, len(STANDARD_CHAR_LIST))

    def test_distributions_sum_to_one(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        row_sums = result.distributions.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestMockOCRReproducibility:
    """The mock must be deterministic given a fixed seed (NFR: reproducibility)."""

    def test_same_seed_same_output(self, blank_image):
        ocr1 = MockOCRAdapter(error_rate=0.2, rng=random.Random(42))
        ocr2 = MockOCRAdapter(error_rate=0.2, rng=random.Random(42))
        r1 = ocr1.predict(blank_image)
        r2 = ocr2.predict(blank_image)
        assert r1.raw_string == r2.raw_string
        assert r1.recovered_code == r2.recovered_code

    def test_different_seeds_different_output(self, blank_image):
        ocr1 = MockOCRAdapter(error_rate=0.5, rng=random.Random(1))
        ocr2 = MockOCRAdapter(error_rate=0.5, rng=random.Random(2))
        # Generate several reads — at least one should differ.
        outputs1 = [ocr1.predict(blank_image).raw_string for _ in range(5)]
        outputs2 = [ocr2.predict(blank_image).raw_string for _ in range(5)]
        assert outputs1 != outputs2


class TestMockOCRWithErrors:
    """With errors injected, the mock exercises the recovery pipeline."""

    def test_low_error_rate_mostly_recovers(self, blank_image):
        # 1% error per char, 100 trials. Almost all should succeed (single-edit
        # corruptions are easy to recover given confidence info).
        ocr = MockOCRAdapter(error_rate=0.01, rng=random.Random(0))
        successes = sum(ocr.predict(blank_image).succeeded for _ in range(100))
        assert successes >= 95

    def test_recovery_edits_tracked(self, blank_image):
        # Force a known code so we can predict the corruption pattern.
        ocr = MockOCRAdapter(
            error_rate=1.0,    # every position corrupted
            rng=random.Random(0),
            true_code_override="CSQU3054383",
        )
        result = ocr.predict(blank_image)
        # When everything is corrupted, recovery may or may not succeed
        # depending on whether the corruptions happen to land on a valid code.
        # Either way, recovery_edits should be reported when recovery succeeded.
        if result.succeeded:
            assert result.recovery_edits is not None
            assert result.recovery_edits >= 0

    def test_true_code_override_used(self, blank_image):
        # With error_rate=0 and override, we should recover the override every time.
        ocr = MockOCRAdapter(
            error_rate=0.0,
            rng=random.Random(0),
            true_code_override="CSQU3054383",
        )
        for _ in range(10):
            result = ocr.predict(blank_image)
            assert result.recovered_code == "CSQU3054383"


class TestMockOCRStructure:
    def test_bounding_box_present(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert isinstance(result.bounding_box, BoundingBox)
        assert result.bounding_box.confidence > 0

    def test_latency_recorded(self, blank_image):
        ocr = MockOCRAdapter(error_rate=0.0, rng=random.Random(0))
        result = ocr.predict(blank_image)
        assert result.latency_ms >= 0  # >=0 to be lenient on fast machines
        assert result.latency_ms < 1000   # mock should be well under 1 second

    def test_name_property(self):
        ocr = MockOCRAdapter(error_rate=0.05)
        assert "Mock" in ocr.name
        assert "0.05" in ocr.name


class TestRealOCRStub:
    def test_real_adapter_raises_until_integrated(self):
        with pytest.raises(NotImplementedError):
            RealOCRAdapter("nonexistent_detector.pt", "nonexistent_transcriber.pt")


class TestErrorTypes:
    """Mock should produce errors that respect character class (letters/digits)."""

    def test_letters_corrupted_to_letters(self, blank_image):
        ocr = MockOCRAdapter(
            error_rate=1.0,
            rng=random.Random(0),
            true_code_override="CSQU3054383",
        )
        # With every position corrupted, check that letter positions still get letters.
        for _ in range(20):
            result = ocr.predict(blank_image)
            # Positions 0-2 must be letters.
            for i in range(3):
                assert result.raw_string[i].isalpha(), (
                    f"Position {i} should be letter, got '{result.raw_string[i]}'"
                )
            # Position 3 must be in U/J/Z.
            assert result.raw_string[3] in "UJZ"
            # Positions 4-10 must be digits.
            for i in range(4, 11):
                assert result.raw_string[i].isdigit(), (
                    f"Position {i} should be digit, got '{result.raw_string[i]}'"
                )