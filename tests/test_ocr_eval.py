"""Tests for FR1 scoring helpers."""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion import FrameRead
from src.ocr_eval import Tally, at_level, best_at_level, char_accuracy, edit_distance, outcome

A, B = "EMCU6461774", "EMCU6461779"


def fr(status, code=None, rotation=0, conf=0.5):
    return FrameRead("558", 0, status, code, None, 0, conf, rotation)


class TestOutcome:
    def test_all_outcomes(self):
        assert outcome(A, A) == "correct"
        assert outcome(B, A) == "wrong"
        assert outcome(None, A) == "missed"
        assert outcome(A, None) == "false_code"
        assert outcome(None, None) == "correct_none"

    def test_char_accuracy(self):
        assert char_accuracy(A, A) == 1.0
        assert abs(char_accuracy(B, A) - 10 / 11) < 1e-9
        assert char_accuracy("X", A) == 0.0                 # floored
        assert edit_distance("EMCU", "EMU") == 1


class TestLevels:
    def test_level_turns_weaker_statuses_into_failures(self):
        reads = [fr("verified", A), fr("recovered", B), fr("inferred", A)]
        assert [r.status for r in at_level(reads, "raw")] == ["verified", "failed", "failed"]
        assert [r.code for r in at_level(reads, "recovered")] == [A, B, None]
        assert [r.status for r in at_level(reads, "inferred")] == ["verified", "recovered", "inferred"]

    def test_best_rotation_chosen_after_level(self):
        # A recovered read at 270 beats an inferred one at 0 at every level that allows it,
        # but at "raw" neither counts.
        rotations = [fr("inferred", B, 0, conf=0.9), fr("recovered", A, 270)]
        assert best_at_level(rotations, "recovered").code == A
        assert best_at_level(rotations, "raw").status == "failed"


class TestTally:
    def test_rates_split_container_and_no_container(self):
        t = Tally()
        for pred, truth in [(A, A), (B, A), (None, A), (A, None), (None, None)]:
            t.add(pred, truth)
        assert t.with_container == 3 and t.without_container == 2
        assert t.rate("correct") == 1 / 3 and t.rate("false_code") == 1 / 2
        assert len(t.char_accs) == 2                         # only where code + label exist

    def test_empty_rates_are_none(self):
        assert Tally().rate("correct") is None and Tally().mean_char_accuracy is None
