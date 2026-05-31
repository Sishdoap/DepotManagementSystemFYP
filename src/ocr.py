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


# --- Real implementation: CCLN + PaddleOCR ---

class RealOCRAdapter(OCRAdapter):
    """Two-stage pipeline: CCLN bounding-box detection + PaddleOCR recognition.

    Loads both models once at construction. predict() runs them in sequence on
    a single image. The adapter accepts ndarray (the existing OCRAdapter
    contract) and converts internally, so the simulator interface is unchanged.

    Recovery: this adapter uses the OCR-confusion-aware substitution recovery
    from your existing predict_paddle.py pipeline (correct_to_valid_iso6346),
    not the beam-search recovery in iso6346.py. PaddleOCR doesn't expose
    per-character probability distributions, so distribution-guided recovery
    is not applicable to this pipeline.
    """

    def __init__(
        self,
        ccln_weights_path: str = "models/ccln.pth",
        *,
        device: str = "cpu",
        paddle_gpu: bool = False,
        use_localization: bool = True,
        try_rotations: bool = True,
    ):
        """
        Args:
            ccln_weights_path: path to the CCLN .pth weights.
            device: 'cpu' or 'cuda' for CCLN.
            paddle_gpu: True to run PaddleOCR on GPU.
            use_localization: if False, skip CCLN and pass the whole image to
                PaddleOCR. Useful when input is already a tight crop or when
                CCLN's training distribution doesn't match input (e.g.,
                synthetic images).
            try_rotations: if True, try 0/90/180/270° and pick the best.
                Slower but recovers vertical/upside-down codes.
        """
        # Imports are local because torch/paddle are heavy and we only want
        # to pay the import cost when the real adapter is actually used.
        import os
        # Paddle 3.x crashes with the new PIR executor + oneDNN; disable both
        # before importing paddle.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")

        import torch
        from src.CCLN import load_ccln

        self._device = torch.device(device)
        self._ccln = load_ccln(ccln_weights_path, self._device)
        self._paddle = self._init_paddle(paddle_gpu)
        self._use_localization = use_localization
        self._try_rotations = try_rotations
        self._ccln_weights_path = ccln_weights_path

    @property
    def name(self) -> str:
        loc = "with_loc" if self._use_localization else "no_loc"
        rot = "rot" if self._try_rotations else "norot"
        return f"RealOCRAdapter({self._device}, {loc}, {rot})"

    @staticmethod
    def _init_paddle(use_gpu: bool):
        from paddleocr import PaddleOCR
        return PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="gpu" if use_gpu else "cpu",
            enable_mkldnn=False,
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
            text_det_unclip_ratio=2.0,
        )

    def predict(self, image) -> OCRResult:
        """Run the full pipeline on one image.

        Args:
            image: PIL.Image.Image (preferred) or (H, W, 3) uint8 ndarray.
                The adapter accepts both for compatibility — PIL is preferred
                because that's how real images are loaded; ndarray is kept
                for synthetic-image fallback.
        """
        import time
        from PIL import Image as PILImage

        start = time.perf_counter()

        # Normalize input to PIL.
        if isinstance(image, np.ndarray):
            pil = PILImage.fromarray(image)
        elif isinstance(image, PILImage.Image):
            pil = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Stage 1: localization (optional).
        bbox_px, crop = self._localize_and_crop(pil)

        # Stage 2: recognition with rotation search.
        code, valid, fragments, rotation = self._recognize(crop)

        # Build OCRResult. We don't have per-position distributions from
        # PaddleOCR, so several fields stay None.
        bbox = BoundingBox(
            x=bbox_px[0],
            y=bbox_px[1],
            width=bbox_px[2] - bbox_px[0],
            height=bbox_px[3] - bbox_px[1],
            confidence=1.0,        # CCLN doesn't return a confidence
        )

        # Mean fragment confidence as a confidence proxy.
        mean_conf = (
            sum(c for _, c in fragments) / len(fragments)
            if fragments
            else 0.0
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return OCRResult(
            raw_string=code,
            distributions=None,           # PaddleOCR doesn't expose these
            char_list=STANDARD_CHAR_LIST,
            bounding_box=bbox,
            recovered_code=code if valid else None,
            is_valid=valid,
            recovery_edits=None,          # adapter doesn't track this granularly
            log_probability=float(np.log(mean_conf)) if mean_conf > 0 else None,
            latency_ms=latency_ms,
        )

    # ---- internals (port of predict_paddle.py logic, adapted for in-process use) ----

    def _localize_and_crop(self, pil):
        """Run CCLN to crop the code region. Returns (bbox_pixels, cropped_pil)."""
        if not self._use_localization:
            W, H = pil.size
            return (0, 0, W, H), pil
        bbox_px = self._localize(pil)
        return bbox_px, pil.crop(bbox_px)

    def _localize(self, pil, pad_frac=0.08):
        import torch
        from src.CCLN import CCLN_TRANSFORM

        W, H = pil.size
        with torch.no_grad():
            tensor = CCLN_TRANSFORM(pil).unsqueeze(0).to(self._device)
            bbox = self._ccln(tensor).squeeze(0).cpu().numpy()

        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))])
        y1, y2 = sorted([max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))])

        # Extra horizontal padding for vertical codes.
        w = x2 - x1
        h = y2 - y1
        is_vertical = h > w * 1.5
        pad_x = pad_frac * 4 if is_vertical else pad_frac
        pad_y = pad_frac
        x1 = max(0.0, x1 - w * pad_x)
        x2 = min(1.0, x2 + w * pad_x)
        y1 = max(0.0, y1 - h * pad_y)
        y2 = min(1.0, y2 + h * pad_y)
        return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

    def _recognize(self, crop_pil):
        rotations = [0, 90, 180, 270] if self._try_rotations else [0]
        best = ("", False, [], 0, -1.0)

        for rot in rotations:
            rotated = crop_pil if rot == 0 else crop_pil.rotate(rot, expand=True)
            fragments = self._ocr_one(rotated)
            code, valid, score = self._best_from_fragments(fragments)
            if score > best[4]:
                best = (code, valid, fragments, rot, score)
            if valid:
                break

        return best[0], best[1], best[2], best[3]

    def _ocr_one(self, crop_pil):
        arr = np.array(crop_pil)
        if hasattr(self._paddle, "predict"):
            raw = self._paddle.predict(arr)
        else:
            raw = self._paddle.ocr(arr, cls=True)
        return list(_flatten_paddle_result(raw))

    @staticmethod
    def _best_from_fragments(fragments):
        from .iso6346_recovery import (
            correct_to_valid_iso6346,
            is_valid_container_code,
            CONTAINER_RE,
        )
        import re

        cleaned = [(re.sub(r"[^A-Z0-9]", "", t.upper()), c) for t, c in fragments]
        cleaned = [(t, c) for t, c in cleaned if t]
        if not cleaned:
            return ("", False, 0.0)

        joined = "".join(t for t, _ in cleaned)
        mean_conf = sum(c for _, c in cleaned) / len(cleaned)

        candidates = []
        for t, _ in cleaned:
            for m in CONTAINER_RE.finditer(t):
                candidates.append(m.group())
        for m in CONTAINER_RE.finditer(joined):
            candidates.append(m.group())

        # Tier 1: any regex hit that already passes ISO 6346.
        for code in candidates:
            if is_valid_container_code(code):
                return (code, True, 1000.0 + mean_conf)

        # Tier 1.5: OCR-confusion correction.
        for code in candidates:
            corrected = correct_to_valid_iso6346(code)
            if corrected:
                return (corrected, True, 900.0 + mean_conf)

        if len(joined) >= 11:
            for i in range(len(joined) - 10):
                corrected = correct_to_valid_iso6346(joined[i : i + 11])
                if corrected:
                    return (corrected, True, 800.0 + mean_conf)

        # Tier 2: regex match without check digit.
        if candidates:
            return (candidates[0], False, mean_conf)

        # Tier 3: longest cleaned fragment.
        return (max((t for t, _ in cleaned), key=len), False, mean_conf * 0.1)


def _flatten_paddle_result(result):
    """Yield (text, confidence). Handles PaddleOCR 2.x and 3.x output shapes."""
    if not result:
        return
    first = result[0]
    if isinstance(first, dict):
        for text, score in zip(first.get("rec_texts", []), first.get("rec_scores", [])):
            yield text, float(score)
    else:
        for line in first or []:
            if not line:
                continue
            text, conf = line[1]
            yield text, float(conf)