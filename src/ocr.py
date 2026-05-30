"""OCR adapter interface for ISO 6346 container code recognition.

Defines a single abstract class `OCRAdapter` with one method, `predict()`.
Two concrete implementations live below:

    MockOCRAdapter — generates synthetic distributions for testing and
        simulation without needing a real model or images.
    RealOCRAdapter — wraps the FYP1 detector + transcriber. Stub here;
        you'll fill in the model loading and inference calls when we
        plug your trained weights in (Component 4.5).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .iso6346 import (
    generate_valid_code,
    recover_with_distributions,
    validate,
)


# Standard OCR character vocabulary: digits, uppercase, lowercase, space.
# Length 63. Matches the model architecture in your collaborator's snippet.
STANDARD_CHAR_LIST: list[str] = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + [" "]
)


@dataclass(frozen=True)
class BoundingBox:
    """A detection from the bounding-box model, in pixel coordinates."""
    x: int
    y: int
    width: int
    height: int
    confidence: float    # detector's confidence in this region being container code


@dataclass(frozen=True)
class OCRResult:
    """Full output of an OCR run on one input image.

    Attributes:
        raw_string: the transcriber's argmax decode, length 11 in the
            successful case but may be any length if the CTC decoder
            emitted a different number of characters.
        distributions: (N, C) array of per-position softmax outputs at the
            timesteps that emitted each character. N is len(raw_string),
            C is len(char_list). None if the transcriber failed.
        char_list: the model's class vocabulary, length C.
        bounding_box: detector's bounding box around the code region.
            None if no detection (transcription was not attempted).
        recovered_code: post-recovery ISO 6346 code if recovery succeeded.
            None if no valid code was recoverable.
        is_valid: True if recovered_code passes ISO 6346 validation.
        recovery_edits: number of character substitutions made during
            recovery. 0 means the raw OCR was already valid. None if
            recovery wasn't attempted (e.g. distribution shape mismatch).
        log_probability: joint log-prob of recovered_code under the OCR
            distributions. Higher (less negative) means higher confidence.
        latency_ms: wall-clock time for this OCR call. Useful for the
            "sub-second per arrival" performance NFR.
    """
    raw_string: str
    distributions: Optional[np.ndarray]
    char_list: list[str]
    bounding_box: Optional[BoundingBox]
    recovered_code: Optional[str]
    is_valid: bool
    recovery_edits: Optional[int]
    log_probability: Optional[float]
    latency_ms: float

    @property
    def succeeded(self) -> bool:
        """True if a valid ISO 6346 code was produced (with or without recovery)."""
        return self.is_valid and self.recovered_code is not None


class OCRAdapter(ABC):
    """Abstract interface every OCR backend must implement."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> OCRResult:
        """Run the full detector + transcriber pipeline on one image.

        Args:
            image: (H, W, 3) uint8 RGB array. Single frame from a camera or
                synthetic generator.

        Returns:
            OCRResult with the recognized code and metadata. On failure
            (no detection, garbled transcription, etc.) `succeeded` will be
            False but the result is still returned for logging.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the adapter (for logging / dashboard)."""


# --- Mock implementation for testing & simulation ---

class MockOCRAdapter(OCRAdapter):
    """Generates synthetic OCR outputs without running any real model.

    Used in two contexts:
    1. Unit tests of downstream components (router, dashboard, database)
       that need an OCR but don't want a dependency on real models or GPUs.
    2. The simulator before real models are plugged in — produces realistic
       results with configurable error rates so we can stress-test recovery.

    The mock fakes the model's output: it picks a "true" code, builds a
    distribution that peaks on that code at each position, then optionally
    corrupts the argmax at some positions to simulate detection errors.
    """

    def __init__(
        self,
        *,
        error_rate: float = 0.0,
        peak_confidence: float = 0.95,
        rng: Optional[random.Random] = None,
        char_list: Optional[list[str]] = None,
        true_code_override: Optional[str] = None,
    ):
        """Configure mock behavior.

        Args:
            error_rate: probability of corrupting any single character.
                0.0 = perfect OCR, 0.1 = ~1 wrong char per 11-char read.
            peak_confidence: probability mass on the "correct" character at
                each position. Lower = less confident model. Real OCR usually
                ranges from 0.85 (hard cases) to 0.99+ (clean cases).
            rng: a random.Random for reproducibility. Tests pass a seeded
                instance; the simulator passes its global rng.
            char_list: vocabulary to use. Defaults to STANDARD_CHAR_LIST.
            true_code_override: if set, every predict() call uses this code
                as the ground truth instead of generating a random one. Useful
                for end-to-end tests where the caller wants to know what
                should be recognized.
        """
        self._error_rate = error_rate
        self._peak = peak_confidence
        self._rng = rng or random.Random()
        self._char_list = char_list or STANDARD_CHAR_LIST
        self._true_code_override = true_code_override

    @property
    def name(self) -> str:
        return f"MockOCRAdapter(error_rate={self._error_rate})"

    def predict(self, image: np.ndarray) -> OCRResult:
        import time
        start = time.perf_counter()

        # 1. Pick the "true" code that's supposedly in the image.
        true_code = self._true_code_override or generate_valid_code(self._rng)

        # 2. Build a per-position distribution peaking on each true character.
        C = len(self._char_list)
        tail_prob = (1.0 - self._peak) / (C - 1)
        distributions = np.full((11, C), tail_prob, dtype=np.float32)
        for pos, ch in enumerate(true_code):
            idx = self._char_list.index(ch)
            distributions[pos, idx] = self._peak

        # 3. Simulate per-character OCR errors by flipping the argmax at some
        #    positions. We shift probability mass from the true char to a
        #    different legal char at that position — this gives a realistic
        #    "wrong but confident" or "ambiguous" distribution.
        for pos in range(11):
            if self._rng.random() < self._error_rate:
                # Pick a wrong character of the SAME TYPE (letter→letter,
                # digit→digit). Realistic OCR errors respect character class.
                self._corrupt_position(distributions, pos, true_code[pos])

        # 4. Reconstruct the OCR's argmax string (what the model "outputs").
        argmax_idx = distributions.argmax(axis=1)
        raw_string = "".join(self._char_list[i] for i in argmax_idx)

        # 5. Run recovery (same pipeline a real adapter would call).
        recovery = recover_with_distributions(distributions, self._char_list)

        # 6. Fake a bounding box (the mock doesn't actually see the image).
        bbox = BoundingBox(
            x=10, y=10, width=200, height=40, confidence=0.99
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return OCRResult(
            raw_string=raw_string,
            distributions=distributions,
            char_list=self._char_list,
            bounding_box=bbox,
            recovered_code=recovery.recovered,
            is_valid=recovery.recovered is not None and validate(recovery.recovered).is_valid,
            recovery_edits=None if recovery.recovered is None else _edit_distance(raw_string, recovery.recovered),
            log_probability=recovery.log_probability if recovery.recovered else None,
            latency_ms=latency_ms,
        )

    def _corrupt_position(self, distributions: np.ndarray, pos: int, true_char: str) -> None:
        """Swap probability mass from the true char to a wrong char of the same type."""
        if true_char.isdigit():
            candidates = [c for c in "0123456789" if c != true_char]
        elif pos == 3:
            candidates = [c for c in "UJZ" if c != true_char]
        else:
            candidates = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != true_char]

        wrong_char = self._rng.choice(candidates)
        true_idx = self._char_list.index(true_char)
        wrong_idx = self._char_list.index(wrong_char)
        # Swap the probabilities so wrong_char wins argmax but truth still has
        # significant mass — this is what realistic OCR confusions look like.
        distributions[pos, true_idx], distributions[pos, wrong_idx] = (
            distributions[pos, wrong_idx],
            distributions[pos, true_idx],
        )


def _edit_distance(a: str, b: str) -> int:
    """Hamming distance between two equal-length strings, or len(b) if unequal."""
    if len(a) != len(b):
        return len(b)
    return sum(1 for x, y in zip(a, b) if x != y)


# --- Real implementation (stub; populated when we plug in your models) ---

class RealOCRAdapter(OCRAdapter):
    """Wraps the trained two-stage pipeline: detector + transcriber.

    Loaded once at startup; predict() runs both models. The actual model
    loading and inference code goes in __init__ and predict() when we
    integrate your weights — currently this is a placeholder that raises.
    """

    def __init__(
        self,
        detector_weights_path: str,
        transcriber_weights_path: str,
        device: str = "cpu",
        char_list: Optional[list[str]] = None,
    ):
        self._detector_path = detector_weights_path
        self._transcriber_path = transcriber_weights_path
        self._device = device
        self._char_list = char_list or STANDARD_CHAR_LIST

        # TODO (Component 4.5): load your two models here.
        # self._detector = load_detector(detector_weights_path, device)
        # self._transcriber = load_transcriber(transcriber_weights_path, device)
        raise NotImplementedError(
            "RealOCRAdapter is a stub. Implementation will be added when "
            "model weights are integrated (Component 4.5)."
        )

    @property
    def name(self) -> str:
        return f"RealOCRAdapter(device={self._device})"

    def predict(self, image: np.ndarray) -> OCRResult:
        raise NotImplementedError("See __init__ TODO.")