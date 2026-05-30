"""ISO 6346 container code validation and confidence-guided recovery.

An ISO 6346 code is 11 characters:
  - Positions 0-2: owner code (3 uppercase letters)
  - Position 3:    category identifier (U, J, or Z)
  - Positions 4-9: serial number (6 digits)
  - Position 10:   check digit (1 digit, computed from positions 0-9)

Check digit algorithm:
  Each letter maps to a numeric value (10 onward, skipping multiples of 11).
  Each of the 10 characters is multiplied by 2**position, summed, mod 11, mod 10.
"""

from dataclasses import dataclass
from itertools import combinations, product
from typing import Iterable
import heapq
import math
import numpy as np


# --- Letter-to-value table (ISO 6346 spec) ---
# Values start at 10 and skip 11, 22, 33 (multiples of 11)
def _build_letter_values() -> dict[str, int]:
    values: dict[str, int] = {}
    skip = {11, 22, 33}
    v = 10
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while v in skip:
            v += 1
        values[ch] = v
        v += 1
    return values


LETTER_VALUES = _build_letter_values()
VALID_CATEGORIES = frozenset({"U", "J", "Z"})


def _char_value(ch: str) -> int:
    """Return numeric value of a character per ISO 6346."""
    if ch.isalpha():
        return LETTER_VALUES[ch]
    return int(ch)


def compute_check_digit(code10: str) -> int:
    """Compute the check digit for the first 10 characters of an ISO 6346 code.

    Raises ValueError if input is malformed.
    """
    if len(code10) != 10:
        raise ValueError(f"Expected 10 characters, got {len(code10)}")
    code10 = code10.upper()
    if not code10[:4].isalpha() or not code10[4:].isdigit():
        raise ValueError(f"Malformed code prefix: {code10}")

    total = sum(_char_value(ch) * (2 ** i) for i, ch in enumerate(code10))
    return (total % 11) % 10


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a candidate ISO 6346 code."""
    code: str
    is_valid: bool
    reason: str = ""    # populated when is_valid is False


def validate(code: str) -> ValidationResult:
    """Validate a complete 11-character ISO 6346 code."""
    if not isinstance(code, str):
        return ValidationResult(str(code), False, "not a string")

    code = code.strip().upper()

    if len(code) != 11:
        return ValidationResult(code, False, f"length {len(code)} != 11")
    if not code[:4].isalpha():
        return ValidationResult(code, False, "first 4 chars must be letters")
    if code[3] not in VALID_CATEGORIES:
        return ValidationResult(code, False, f"category {code[3]} not in U/J/Z")
    if not code[4:].isdigit():
        return ValidationResult(code, False, "last 7 chars must be digits")

    expected = compute_check_digit(code[:10])
    actual = int(code[10])
    if expected != actual:
        return ValidationResult(
            code, False, f"check digit {actual} != expected {expected}"
        )
    return ValidationResult(code, True)


# --- Confidence-guided recovery ---

# Candidate substitutions for each character position type.
# Letters in positions 0-2 can only be replaced by letters; position 3 only by U/J/Z;
# positions 4-10 only by digits. We allow ANY character of the right type as a
# candidate (not just visually-similar pairs) because the OCR confidence vector
# already tells us which alternatives the model considered plausible.
def _candidates_for_position(pos: int) -> str:
    if pos < 3:
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if pos == 3:
        return "UJZ"
    return "0123456789"


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of attempting to recover a valid code from a noisy OCR read."""
    original: str
    recovered: str | None
    edits: int                 # number of character substitutions made
    candidates_tried: int      # how many combinations we evaluated

    @property
    def succeeded(self) -> bool:
        return self.recovered is not None


def recover(
    code: str,
    char_confidences: list[float] | None = None,
    max_edits: int = 2,
) -> RecoveryResult:
    """Attempt to recover a valid ISO 6346 code from a noisy OCR output.

    Strategy: try substituting up to `max_edits` characters, prioritising
    positions with the lowest OCR confidence. Return the first substitution
    set that produces a code passing full ISO 6346 validation.

    IMPORTANT: this returns *a* valid code within edit distance, not
    necessarily the *original* code. The ISO 6346 check digit is mod 10, so
    roughly 1 in 10 random single-character substitutions yields a valid
    code by chance. Recovery is reliable only when (a) confidence information
    strongly constrains which positions are corrupted, and (b) corruptions
    are concentrated at a few specific positions. With weak or absent
    confidence information, the recovered code may differ from the original.

    For higher-fidelity recovery, supply per-position character probability
    distributions (see future `recover_with_distributions` extension).

    Args:
        code: the OCR-predicted 11-character string (case-insensitive).
        char_confidences: optional list of 11 floats in [0,1]; lower means
            less confident in that character. If None, all positions are
            considered equally likely to be wrong.
        max_edits: maximum number of characters to substitute. Default 2
            keeps the search tractable while catching most realistic errors.

    Returns:
        RecoveryResult. recovered=None if no valid code found within budget.
    """
    code = code.strip().upper()
    if len(code) != 11:
        return RecoveryResult(code, None, 0, 0)

    # If already valid, no recovery needed.
    if validate(code).is_valid:
        return RecoveryResult(code, code, 0, 1)

    # Determine search order: lowest-confidence positions first.
    if char_confidences is None:
        char_confidences = [0.5] * 11
    if len(char_confidences) != 11:
        raise ValueError("char_confidences must have length 11")

    positions_ranked = sorted(range(11), key=lambda i: char_confidences[i])

    tried = 0
    # Try edit_count = 1, 2, ..., max_edits (smallest fixes first).
    for edit_count in range(1, max_edits + 1):
        # Choose which positions to edit, prioritising low-confidence ones.
        # We limit the combinatorial search to the K lowest-confidence positions
        # where K = max(edit_count + 2, 4) — this is a search-budget heuristic.
        search_window = positions_ranked[: max(edit_count + 2, 4)]

        for positions in combinations(search_window, edit_count):
            # Generate Cartesian product of candidate characters for each chosen position.
            candidate_chars = [_candidates_for_position(p) for p in positions]
            # Exclude the original character at each chosen position
            # (substituting a char with itself isn't an edit).
            candidate_chars = [
                "".join(c for c in chars if c != code[p])
                for p, chars in zip(positions, candidate_chars)
            ]

            for replacement in product(*candidate_chars):
                tried += 1
                chars = list(code)
                for p, new_ch in zip(positions, replacement):
                    chars[p] = new_ch
                candidate = "".join(chars)
                if validate(candidate).is_valid:
                    return RecoveryResult(code, candidate, edit_count, tried)

    return RecoveryResult(code, None, 0, tried)


# --- Synthesis (useful for tests and simulation) ---

def generate_valid_code(rng) -> str:
    """Generate a syntactically and check-digit-valid ISO 6346 code.

    Args:
        rng: a random.Random instance (passed in for reproducibility).
    """
    owner = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    category = rng.choice("UJZ")
    serial = "".join(rng.choices("0123456789", k=6))
    prefix = owner + category + serial
    return prefix + str(compute_check_digit(prefix))

# --- Positional constraints (ISO 6346 hard rules) ---

# For each position, the set of legal characters as a string.
POSITION_ALPHABETS: tuple[str, ...] = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",   # 0: owner code letter
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",   # 1: owner code letter
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",   # 2: owner code letter
    "UJZ",                          # 3: equipment category
    "0123456789",                   # 4-9: serial number
    "0123456789",
    "0123456789",
    "0123456789",
    "0123456789",
    "0123456789",
    "0123456789",                   # 10: check digit (but we compute, not search)
)


def build_position_mask(
    position: int, char_list: list[str]
) -> np.ndarray:
    """Return a 0/1 mask over `char_list` selecting only legal chars at `position`.

    Args:
        position: 0..10.
        char_list: the OCR model's full class list (e.g. 63 entries).

    Returns:
        np.ndarray of shape (len(char_list),) with 1.0 at legal class indices.
    """
    legal = set(POSITION_ALPHABETS[position])
    mask = np.zeros(len(char_list), dtype=np.float32)
    for i, ch in enumerate(char_list):
        if ch.upper() in legal:
            mask[i] = 1.0
    return mask


def masked_renormalized(
    distribution: np.ndarray, mask: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Apply mask and renormalize to a proper probability distribution."""
    masked = distribution * mask
    total = masked.sum()
    if total < eps:
        # No probability mass on any legal character — fall back to uniform over legals.
        legal_count = mask.sum()
        if legal_count == 0:
            raise ValueError("mask has no legal characters")
        return mask / legal_count
    return masked / total


# --- Beam-search recovery using full per-position distributions ---

@dataclass(frozen=True)
class DistributionRecoveryResult:
    """Outcome of distribution-guided recovery."""
    original: str
    recovered: str | None
    log_probability: float       # joint log-prob of recovered positions 0-9
    candidates_evaluated: int

    @property
    def succeeded(self) -> bool:
        return self.recovered is not None


def recover_with_distributions(
    distributions: np.ndarray,
    char_list: list[str],
    beam_width: int = 20,
) -> DistributionRecoveryResult:
    """Recover a valid ISO 6346 code via beam search over per-position distributions.

    Scoring (Bayesian): each candidate code is scored by the joint log-probability
    of all 11 positions under the OCR's per-position distributions, where position
    10 uses the OCR's probability of the *computed* check digit (not the observed
    character at position 10). The check digit at position 10 is determined by
    positions 0-9 via the ISO 6346 algorithm; the OCR's reading of position 10 is
    a soft signal, not a hard filter.

    Tie-breaker: when two candidates have equal total joint log-probability,
    prefer the one with higher joint over positions 0-9 alone. Positions 0-9
    carry semantic content; position 10 is a single derived digit.

    Args:
        distributions: (11, C) numpy array. distributions[i] is the softmax
            over classes at the timestep that emitted output position i.
        char_list: the OCR model's class vocabulary, length C.
        beam_width: beams kept at each step of the search. 10-20 recommended.

    Returns:
        DistributionRecoveryResult with the highest-scoring valid code, or
        recovered=None if no valid code was in the beam.
    """
    if distributions.shape[0] != 11:
        return DistributionRecoveryResult(
            original="",
            recovered=None,
            log_probability=float("-inf"),
            candidates_evaluated=0,
        )
    if distributions.shape[1] != len(char_list):
        raise ValueError(
            f"distribution width {distributions.shape[1]} != "
            f"char_list length {len(char_list)}"
        )

    # Reconstruct the OCR's argmax read (for the `original` field).
    argmax_indices = distributions.argmax(axis=1)
    original = "".join(char_list[i] for i in argmax_indices).upper()

    # Masked + renormalized distributions per position.
    masks = [build_position_mask(p, char_list) for p in range(11)]
    masked_dists = [
        masked_renormalized(distributions[p], masks[p]) for p in range(11)
    ]

    # Precompute log-prob of each digit at position 10 (for scoring computed
    # check digits). We use the masked distribution: only digits are legal.
    pos10_logprob: dict[str, float] = {}
    for digit in "0123456789":
        idx = char_list.index(digit)
        p = masked_dists[10][idx]
        pos10_logprob[digit] = math.log(p) if p > 0 else float("-inf")

    # Per-position top-K candidates for positions 0-9, with log-probabilities.
    K = beam_width
    per_position_topk: list[list[tuple[str, float]]] = []
    for p in range(10):
        dist = masked_dists[p]
        top_idx = np.argsort(dist)[::-1][:K]
        entries: list[tuple[str, float]] = []
        for idx in top_idx:
            prob = dist[idx]
            if prob <= 0:
                continue
            entries.append((char_list[idx].upper(), math.log(prob)))
        per_position_topk.append(entries)

    # Beam search across positions 0-9.
    # Each beam: (cumulative_log_prob_0to9, partial_string)
    beams: list[tuple[float, str]] = [(0.0, "")]
    evaluated = 0
    for p in range(10):
        extensions: list[tuple[float, str]] = []
        for cum_lp, partial in beams:
            for ch, lp in per_position_topk[p]:
                extensions.append((cum_lp + lp, partial + ch))
                evaluated += 1
        beams = heapq.nlargest(beam_width, extensions, key=lambda x: x[0])

    # Score each beam by full 11-position joint. When two candidates have
    # near-equal totals (within float precision), prefer the one with higher
    # joint over positions 0-9 — those carry semantic content, while position
    # 10 is a single derived digit.
    SCORE_EPS = 1e-6

    scored: list[tuple[float, float, str]] = []  # (total_lp, lp_0to9, full_code)
    for cum_lp, prefix in beams:
        try:
            computed_cd = compute_check_digit(prefix)
        except ValueError:
            continue
        cd_str = str(computed_cd)
        full_code = prefix + cd_str
        if not validate(full_code).is_valid:
            continue
        total_lp = cum_lp + pos10_logprob.get(cd_str, float("-inf"))
        scored.append((total_lp, cum_lp, full_code))

    if not scored:
        return DistributionRecoveryResult(
            original=original,
            recovered=None,
            log_probability=float("-inf"),
            candidates_evaluated=evaluated,
        )

    # Sort: primary key is total_lp (rounded to suppress float noise),
    # tie-break on lp_0to9. Both descending.
    scored.sort(key=lambda x: (-round(x[0] / SCORE_EPS) * SCORE_EPS, -x[1]))
    best_total, best_lp09, best_code = scored[0]

    return DistributionRecoveryResult(
        original=original,
        recovered=best_code,
        log_probability=best_total,
        candidates_evaluated=evaluated,
    )