"""Tests for ISO 6346 validation and recovery."""

import random
import pytest
import sys
from pathlib import Path
import numpy as np
import math

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.iso6346 import (
    LETTER_VALUES,
    compute_check_digit,
    validate,
    recover,
    generate_valid_code,
    build_position_mask,
    masked_renormalized,
    recover_with_distributions,
    POSITION_ALPHABETS,
)


class TestLetterValues:
    def test_skip_multiples_of_eleven(self):
        # 11, 22, 33 must not appear in the table.
        assert 11 not in LETTER_VALUES.values()
        assert 22 not in LETTER_VALUES.values()
        assert 33 not in LETTER_VALUES.values()

    def test_known_values(self):
        # Spot-checks from the ISO 6346 specification.
        assert LETTER_VALUES["A"] == 10
        assert LETTER_VALUES["B"] == 12     # 11 is skipped
        assert LETTER_VALUES["K"] == 21     # 22 would be next, skipped
        assert LETTER_VALUES["L"] == 23
        assert LETTER_VALUES["Z"] == 38


class TestCheckDigit:
    def test_known_valid_code(self):
        # CSQU3054383 is a canonical example from the ISO spec.
        assert compute_check_digit("CSQU305438") == 3

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            compute_check_digit("TOOSHORT")


class TestValidate:
    def test_canonical_valid_code(self):
        result = validate("CSQU3054383")
        assert result.is_valid
        assert result.reason == ""

    def test_lowercase_accepted(self):
        # Validation should normalise case before checking.
        assert validate("csqu3054383").is_valid

    def test_whitespace_stripped(self):
        assert validate("  CSQU3054383  ").is_valid

    def test_wrong_length(self):
        result = validate("CSQU305438")
        assert not result.is_valid
        assert "length" in result.reason

    def test_invalid_category(self):
        # 'X' is not in {U, J, Z}.
        result = validate("CSQX3054383")
        assert not result.is_valid
        assert "category" in result.reason

    def test_digits_in_letter_positions(self):
        result = validate("12QU3054383")
        assert not result.is_valid

    def test_wrong_check_digit(self):
        # Change last digit to break the check.
        result = validate("CSQU3054384")
        assert not result.is_valid
        assert "check digit" in result.reason


class TestRecover:
    def test_passes_through_valid_code(self):
        result = recover("CSQU3054383")
        assert result.succeeded
        assert result.recovered == "CSQU3054383"
        assert result.edits == 0

    def test_single_digit_corruption_recovered(self):
        # Corrupt the check digit; confidence vector flags it.
        confidences = [0.99] * 10 + [0.10]
        result = recover("CSQU3054384", char_confidences=confidences)
        assert result.succeeded
        assert result.recovered == "CSQU3054383"
        assert result.edits == 1

    def test_recovers_without_confidence_hint(self):
        # No confidence info: brute-force still finds A valid code (not necessarily
        # the original). The mod-10 check digit is weak — many valid codes lie
        # within edit distance 1 of any given corruption.
        result = recover("CSQU3054384")
        assert result.succeeded
        assert validate(result.recovered).is_valid
        assert result.edits == 1

    def test_unrecoverable_within_budget(self):
        # '1234567890Z' violates: positions 0-3 must all be letters (with position 3
        # restricted to U/J/Z), and position 10 must be a digit. Minimum 5 edits
        # required — well past max_edits=2.
        result = recover("1234567890Z")
        assert not result.succeeded
        assert result.recovered is None

    def test_two_digit_corruption_recovered(self):
        # With two corruptions and confidence flagging both, we recover SOME valid
        # code within budget. We cannot guarantee returning the original — many
        # valid codes exist within edit distance 2 of any corruption. This is a
        # fundamental property of the ISO 6346 checksum, not an algorithm flaw.
        rng = random.Random(42)
        code = generate_valid_code(rng)
        corrupted = list(code)
        corrupted[5] = "0" if code[5] != "0" else "1"
        corrupted[8] = "0" if code[8] != "0" else "1"
        corrupted_str = "".join(corrupted)

        confidences = [0.99] * 11
        confidences[5] = 0.05
        confidences[8] = 0.10

        result = recover(corrupted_str, char_confidences=confidences, max_edits=2)
        assert result.succeeded
        assert validate(result.recovered).is_valid
        assert result.edits <= 2


class TestGenerate:
    def test_generated_codes_are_valid(self):
        rng = random.Random(0)
        for _ in range(1000):
            code = generate_valid_code(rng)
            assert validate(code).is_valid

    def test_reproducible_with_seed(self):
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        codes1 = [generate_valid_code(rng1) for _ in range(10)]
        codes2 = [generate_valid_code(rng2) for _ in range(10)]
        assert codes1 == codes2


# Standard char list mimicking the real OCR model: digits, uppercase, lowercase, space.
# (Length 63; ordering doesn't matter as long as build_position_mask uses it consistently.)
STD_CHAR_LIST = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
    + [" "]
)
assert len(STD_CHAR_LIST) == 63


def _one_hot_distributions(code: str, char_list: list[str], peak: float = 0.95) -> np.ndarray:
    """Build a (11, C) distribution that peaks on the given code's characters.

    Useful for constructing test inputs that mimic a high-confidence OCR output.
    `peak` of probability mass goes on the correct character; the rest is spread
    uniformly over the other classes.
    """
    C = len(char_list)
    dists = np.full((11, C), (1.0 - peak) / (C - 1), dtype=np.float32)
    for pos, ch in enumerate(code):
        idx = char_list.index(ch)
        dists[pos, idx] = peak
    return dists


def _ambiguous_distribution(
    code: str, char_list: list[str], pos: int, alt: str, peak: float = 0.45, alt_prob: float = 0.40
) -> np.ndarray:
    """Distribution where position `pos` is split between the correct char and `alt`."""
    dists = _one_hot_distributions(code, char_list, peak=0.95)
    idx_correct = char_list.index(code[pos])
    idx_alt = char_list.index(alt)
    dists[pos] = (1.0 - peak - alt_prob) / (len(char_list) - 2)
    dists[pos, idx_correct] = peak
    dists[pos, idx_alt] = alt_prob
    return dists


class TestPositionMask:
    def test_letter_positions(self):
        mask = build_position_mask(0, STD_CHAR_LIST)
        # 26 uppercase letters should be legal (lowercase masked even though
        # they share characters — mask is on the char_list entry as-is).
        # Note: our mask matches by uppercased character, so lowercase 'a'
        # also passes the membership test. That's intentional — if the OCR
        # emits a lowercase letter at a letter position, we want it counted,
        # and the recovery converts to uppercase at output time.
        assert mask.sum() == 52   # 26 upper + 26 lower

    def test_category_position(self):
        mask = build_position_mask(3, STD_CHAR_LIST)
        # Only U, J, Z (and their lowercase variants in the std char list).
        assert mask.sum() == 6

    def test_digit_positions(self):
        mask = build_position_mask(4, STD_CHAR_LIST)
        assert mask.sum() == 10   # 0-9


class TestMaskedRenormalized:
    def test_sums_to_one(self):
        dist = np.array([0.5, 0.3, 0.1, 0.1])
        mask = np.array([1.0, 0.0, 1.0, 0.0])
        out = masked_renormalized(dist, mask)
        assert math.isclose(out.sum(), 1.0)

    def test_zero_mass_falls_back_to_uniform(self):
        dist = np.array([0.0, 0.0, 0.5, 0.5])
        mask = np.array([1.0, 1.0, 0.0, 0.0])
        out = masked_renormalized(dist, mask)
        assert math.isclose(out.sum(), 1.0)
        assert out[0] == 0.5 and out[1] == 0.5


class TestRecoverWithDistributions:
    def test_clean_high_confidence_input(self):
        rng = random.Random(0)
        code = generate_valid_code(rng)
        dists = _one_hot_distributions(code, STD_CHAR_LIST, peak=0.99)
        result = recover_with_distributions(dists, STD_CHAR_LIST)
        assert result.succeeded
        assert result.recovered == code

    def test_recovers_when_check_digit_misread(self):
        # OCR confidently reads positions 0-9 correctly but mis-reads pos 10.
        rng = random.Random(1)
        code = generate_valid_code(rng)
        wrong_cd = str((int(code[10]) + 1) % 10)
        corrupted = code[:10] + wrong_cd
        dists = _one_hot_distributions(corrupted, STD_CHAR_LIST, peak=0.95)
        result = recover_with_distributions(dists, STD_CHAR_LIST)
        assert result.succeeded
        # Should recover the original code by recomputing check digit.
        assert result.recovered == code

    def test_recovers_when_alt_char_is_correct(self):
        # OCR sees '8' at position 5, with '1' as a close alternative.
        # Truth is '1'. Distribution-guided recovery should pick '1'.
        rng = random.Random(2)
        code = generate_valid_code(rng)
        # Construct a code we know has '1' at position 5.
        code = code[:5] + "1" + code[6:10]
        code += str(compute_check_digit(code))

        # OCR top guess at pos 5 is '8' (wrong), with '1' as runner-up.
        corrupted = code[:5] + "8" + code[6:]
        dists = _one_hot_distributions(corrupted, STD_CHAR_LIST, peak=0.50)
        # Now bias pos 5: '8' at 0.45, '1' at 0.40.
        pos5 = np.full(len(STD_CHAR_LIST), 0.15 / (len(STD_CHAR_LIST) - 2))
        pos5[STD_CHAR_LIST.index("8")] = 0.45
        pos5[STD_CHAR_LIST.index("1")] = 0.40
        dists[5] = pos5

        result = recover_with_distributions(dists, STD_CHAR_LIST)
        assert result.succeeded
        # The check digit at position 10 disambiguates: code with '1' is valid,
        # code with '8' would have a different check digit.
        assert result.recovered == code

    def test_invalid_distribution_shape(self):
        bad = np.zeros((5, 63))
        result = recover_with_distributions(bad, STD_CHAR_LIST)
        assert not result.succeeded

    def test_invalid_char_list_size(self):
        dists = np.zeros((11, 50))
        with pytest.raises(ValueError):
            recover_with_distributions(dists, STD_CHAR_LIST)