"""Tests for container-code reading from OCR fragments.

Most cases are real failure patterns from the FYP1 OCR benchmark.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.iso6346 import validate
from src.iso6346_recovery import (
    FAILED,
    INFERRED,
    RECOVERED,
    VERIFIED,
    ReaderConfig,
    candidate_strings,
    extract_size_type,
    is_valid_container_code,
    read_candidate,
    read_container_code,
    reading_key,
    symmetric_pairs,
)

FORCED_ONLY = ReaderConfig(max_confusions=0)
ALL_CONFUSIONS = ReaderConfig(max_confusions=1)


class TestVerified:
    def test_clean_read(self):
        r = read_container_code([("TEMU", 0.98), ("6372145", 0.99)])
        assert (r.code, r.status, r.edits) == ("TEMU6372145", VERIFIED, 0)
        assert r.is_validated

    def test_plain_strings_accepted(self):
        assert read_container_code(["TEMU6372145"]).code == "TEMU6372145"

    def test_lowercase_and_punctuation_cleaned(self):
        assert read_container_code(["temu-637 2145"]).code == "TEMU6372145"


class TestReordering:
    def test_serial_read_before_owner(self):
        r = read_container_code([("6372145", 0.99), ("TEMU", 0.97)])
        assert (r.code, r.status) == ("TEMU6372145", VERIFIED)

    def test_owner_glued_after_digits(self):
        assert read_container_code(["7773935CCLU"]).code == "CCLU7773935"

    def test_digit_before_owner_moves_to_end(self):
        assert read_container_code(["9TGHU909552"]).code == "TGHU9095529"

    def test_check_digit_read_first_when_enabled(self):
        r = read_container_code(["CSLU7236936"], ReaderConfig(try_check_digit_first=True))
        assert (r.code, r.status) == ("CSLU2369367", VERIFIED)

    def test_check_digit_first_off_by_default(self):
        # Measured to cost more correct reads than it adds.
        assert read_container_code(["CSLU7236936"]).status != VERIFIED

    def test_check_digit_first_only_counts_if_exact(self):
        # The speculative rearrangement must not be combined with other fixes.
        for cand, speculative in candidate_strings(["KMTU9284037"]):
            if speculative:
                assert read_candidate(cand, FORCED_ONLY).status != VERIFIED
        assert read_container_code(["KMTU9284037"], FORCED_ONLY).status != VERIFIED


class TestRoleAssembly:
    """The boxed check digit often comes back as its own fragment."""

    def test_reversed_fragments_at_180_degrees(self):
        # Joining in reading order or moving the owner to the front would
        # give CSNU4670973, which passes the check digit by luck.
        r = read_container_code([("4", 0.9), ("670973", 0.9), ("CSNU", 0.9)])
        assert (r.code, r.status) == ("CSNU6709734", VERIFIED)

    def test_owner_serial_check_in_order(self):
        r = read_container_code(["OFSU", "303535", "0"])
        assert r.code == "OFSU3035350"

    def test_missing_check_fragment_inferred(self):
        r = read_container_code(["CLHU", "866444"])
        assert (r.code, r.status) == ("CLHU8664440", INFERRED)

    def test_two_single_digits_not_assembled(self):
        # Which one is the check digit is unknown, so role assembly declines
        # (the reading-order window is still tried).
        from src.iso6346_recovery import _assembled_by_role
        assert _assembled_by_role(["UACU", "350968", "1", "4"]) == []
        assert _assembled_by_role(["UACU", "350968", "4"]) == ["UACU3509684"]


class TestNoiseFragments:
    @pytest.mark.parametrize("noise,size", [
        ("45G", None), ("4561", "45G1"), ("22C1", "22G1"), ("2ZG1", "22G1"), ("5G1", None),
    ])
    def test_garbled_size_type_fragment_dropped(self, noise, size):
        r = read_container_code(["BMOU", noise, "503287"])
        assert noise not in (r.raw or "")
        assert r.size_type == size

    def test_plate_text_dropped(self):
        r = read_container_code(["2O.KGS", "22G1", "BMOU", "122131", "1"])
        assert (r.code, r.size_type) == ("BMOU1221311", "22G1")


class TestConflicts:
    def test_different_validated_codes_from_two_arrangements_rejected(self):
        # Find a serial where the code and its check-digit-first rearrangement
        # both pass, so two arrangements give two different verified codes.
        for n in range(1_000_000, 1_100_000):
            digits = str(n)
            normal, rotated = "TEMU" + digits, "TEMU" + digits[1:] + digits[0]
            if validate(normal).is_valid and validate(rotated).is_valid and normal != rotated:
                break
        cfg = ReaderConfig(try_check_digit_first=True)
        r = read_container_code([normal], cfg)
        assert r.status == FAILED and "conflict" in r.note
        assert read_container_code([normal], ReaderConfig(try_check_digit_first=True,
                                                          reject_conflicts=False)).code == normal


class TestSizeType:
    def test_glued_size_type_removed_and_reported(self):
        r = read_container_code(["1855130CSNU22G1"])
        assert (r.code, r.status, r.size_type) == ("CSNU1855130", VERIFIED, "22G1")

    def test_separate_size_type_fragment(self):
        r = read_container_code([("TCLU", 0.9), ("5462020", 0.9), ("45G1", 0.9)])
        assert (r.code, r.size_type) == ("TCLU5462020", "45G1")

    def test_serial_digits_not_mistaken_for_size_type(self):
        frags, size = extract_size_type(["DRYU4154476"])
        assert size is None and frags == ["DRYU4154476"]


class TestForcedFixes:
    @pytest.mark.parametrize("read,truth", [
        ("BM0U5075452", "BMOU5075452"),   # digit 0 in owner code -> O
        ("0OLU6517810", "OOLU6517810"),
        ("FC1U3663692", "FCIU3663692"),   # 1 -> I
        ("CZTU000084O", "CZTU0000840"),   # letter O in check digit -> 0
        ("CSNU630241A", "CSNU6302414"),   # A -> 4
    ])
    def test_type_forced_conversion(self, read, truth):
        r = read_container_code([read], FORCED_ONLY)
        assert (r.code, r.status) == (truth, RECOVERED)
        assert r.edits == sum(a != b for a, b in zip(read, truth))


class TestInferred:
    def test_missing_check_digit_is_inferred_not_validated(self):
        r = read_container_code(["MEDU229184"])
        assert (r.code, r.status) == ("MEDU2291844", INFERRED)
        assert not r.is_validated

    def test_size_type_glued_and_check_digit_missing(self):
        r = read_container_code(["CSNU45G1740176"])
        assert (r.code, r.status, r.size_type) == ("CSNU7401767", INFERRED, "45G1")

    def test_wrong_check_digit_recomputed_when_nothing_verifies(self):
        r = read_container_code(["CCLU5265244"], FORCED_ONLY)
        assert (r.code, r.status) == ("CCLU5265241", INFERRED)

    def test_inference_can_be_disabled(self):
        r = read_container_code(["MEDU229184"], ReaderConfig(allow_infer=False))
        assert r.status == FAILED and r.code is None

    def test_ten_char_slice_not_used_when_eleven_were_read(self):
        # The full 11-character read is ambiguous; a 10-character slice of it
        # must not be used to infer a code instead.
        cands = [c for c, _ in candidate_strings(["SNWU1007500"])]
        assert "SNWU100750" not in cands


class TestConfusions:
    def test_unique_confusion_recovered(self):
        cfg = ReaderConfig(max_confusions=1, letter_confusions=symmetric_pairs([("U", "V")]),
                           digit_confusions={})
        r = read_container_code(["TEMV6372145"], cfg)
        assert (r.code, r.status) == ("TEMU6372145", RECOVERED)

    def test_ambiguous_read_rejected_not_guessed(self):
        r = read_container_code(["EMCU3858174"], ALL_CONFUSIONS)
        assert r.status == FAILED and r.code is None
        assert "ambiguous" in r.note

    def test_check_digit_never_substituted_as_recovery(self):
        # Changing only the check digit must yield "inferred", never "recovered".
        r = read_container_code(["TEMU6372146"], ALL_CONFUSIONS)
        assert not (r.status == RECOVERED and r.code[:10] == "TEMU637214")

    def test_category_letter_enforced(self):
        assert read_container_code(["TEMX6372145"], FORCED_ONLY).status != VERIFIED


class TestLengthHandling:
    def test_trailing_junk_ignored(self):
        # Text after a complete code (e.g. the check-digit box read as "1")
        # is outside the code, so the read is still exact.
        r = read_container_code(["TEMU63721451"], FORCED_ONLY)
        assert (r.code, r.status) == ("TEMU6372145", VERIFIED)

    def test_extra_character_inside_code_not_guessed(self):
        # Trying all 12 single-character deletions validated wrong codes about
        # as often as right ones on the benchmark, so it isn't attempted.
        assert not read_container_code(["TEMU63772145"]).is_validated

    def test_too_short_fails(self):
        assert read_container_code(["9951TGHU"]).status == FAILED

    def test_empty_fails(self):
        assert read_container_code([]).status == FAILED


class TestHelpers:
    def test_is_valid_container_code_is_strict(self):
        assert is_valid_container_code("TEMU6372145")
        assert not is_valid_container_code("temu6372145")
        assert not is_valid_container_code(" TEMU6372145")

    def test_reading_key_orders_by_status_then_edits_then_confidence(self):
        verified = read_container_code(["TEMU6372145"])
        inferred = read_container_code(["MEDU229184"])
        assert reading_key(verified, [("x", 0.1)]) > reading_key(inferred, [("x", 0.99)])
        assert reading_key(verified, [("x", 0.9)]) > reading_key(verified, [("x", 0.5)])

    def test_inferred_codes_pass_validation(self):
        r = read_container_code(["UESU475882"])
        assert validate(r.code).is_valid


def test_result_independent_of_hash_seed():
    """The old recovery iterated sets; results must not depend on PYTHONHASHSEED."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src.iso6346_recovery import read_container_code, ReaderConfig\n"
        "cfg = ReaderConfig(max_confusions=2)\n"
        "for s in ['EMCU3858174', 'SNWU1007500', 'CSNU6617814', 'ZSSU7048197']:\n"
        "    r = read_container_code([s], cfg); print(r.code, r.status)\n"
    ) % str(root_dir)
    outputs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.add(subprocess.run([sys.executable, "-c", code], env=env,
                                   capture_output=True, text=True, check=True).stdout)
    assert len(outputs) == 1
