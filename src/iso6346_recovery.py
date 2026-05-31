"""OCR-confusion-aware ISO 6346 recovery.

Port of correct_to_valid_iso6346 from predict_paddle.py. Used by RealOCRAdapter
when PaddleOCR's output fails ISO 6346 validation. Substitutes commonly-confused
character pairs (O↔0, S↔5, etc.) up to a small edit budget.

This is separate from src/iso6346.py's recover_with_distributions(), which
uses per-position softmax probabilities — not available from PaddleOCR.
"""

from __future__ import annotations

import re
from itertools import product

CONTAINER_RE = re.compile(r"[A-Z]{4}\d{7}")

# Common OCR confusions.
LETTER_FROM_DIGIT = {
    "0": "O", "1": "I", "2": "Z", "5": "S",
    "6": "G", "7": "T", "8": "B",
}
DIGIT_FROM_LETTER = {v: k for k, v in LETTER_FROM_DIGIT.items()}
DIGIT_FROM_LETTER.update({"D": "0", "Q": "0", "A": "4"})

# ISO 6346 letter values, skipping multiples of 11.
_LETTER_VAL = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38,
}


def iso6346_check_digit(code10: str) -> int:
    total = 0
    for i, ch in enumerate(code10):
        val = _LETTER_VAL[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    return total % 11 % 10


def is_valid_container_code(code: str) -> bool:
    if not CONTAINER_RE.fullmatch(code):
        return False
    return iso6346_check_digit(code[:10]) == int(code[10])


def _alpha_candidates(ch: str):
    out = {ch}
    if ch in LETTER_FROM_DIGIT:
        out.add(LETTER_FROM_DIGIT[ch])
    return out if ch.isalpha() else (LETTER_FROM_DIGIT.get(ch, ch),)


def _digit_candidates(ch: str):
    out = {ch}
    if ch in DIGIT_FROM_LETTER:
        out.add(DIGIT_FROM_LETTER[ch])
    return out if ch.isdigit() else (DIGIT_FROM_LETTER.get(ch, ch),)


def correct_to_valid_iso6346(text: str, max_subs: int = 3):
    """Substitute up to max_subs OCR-confused chars until checksum passes."""
    if len(text) != 11:
        return None
    slots = []
    for i, ch in enumerate(text):
        if i < 4:
            slots.append(list(_alpha_candidates(ch)))
        else:
            slots.append(list(_digit_candidates(ch)))
    for combo in product(*slots):
        if sum(a != b for a, b in zip(combo, text)) > max_subs:
            continue
        cand = "".join(combo)
        if CONTAINER_RE.fullmatch(cand) and is_valid_container_code(cand):
            return cand
    return None