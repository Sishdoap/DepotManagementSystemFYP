"""Scoring container-code reads against labels (FR1 / FR2 evidence).

Used by scripts/score_fr1.py to build the FR1 results table from the
footage events: per-frame reads, one frame per event, multi-frame fusion
per camera, and cross-camera fusion — each at three trust levels, as the
brief requires raw and recovered accuracy to be reported separately:

    raw        only codes that passed the check digit unchanged (verified)
    recovered  + codes made valid by type-forced fixes (recovered); the
               check digit was still read, so it still confirms them
    inferred   + codes whose check digit was computed (inferred, option B)
               — unverified, so reported as its own level

A level is applied by treating every reading below it as failed before
the per-frame best rotation is chosen, so "raw" really means a pipeline
that only ever outputs verified codes.

Outcome of one prediction against one label:

    correct       a code was output and it equals the label
    wrong         a code was output and it differs (the costly error:
                  a wrong container is booked in)
    missed        the vehicle carries a container but no code was output
    false_code    the vehicle carries no container but a code was output
    correct_none  no container and no code

Character accuracy compares an output code with the label position by
position after alignment (1 - edit distance / 11), only where a code was
output — a missed read has no characters to compare.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

from .fusion import FrameRead, best_reading
from .iso6346_recovery import FAILED, INFERRED, RECOVERED, VERIFIED

LEVELS = {
    "raw": {VERIFIED},
    "recovered": {VERIFIED, RECOVERED},
    "inferred": {VERIFIED, RECOVERED, INFERRED},
}
OUTCOMES = ("correct", "wrong", "missed", "false_code", "correct_none")


def outcome(pred: Optional[str], truth: Optional[str]) -> str:
    """Classify one prediction against one label (see module docstring)."""
    if truth:
        if not pred:
            return "missed"
        return "correct" if pred == truth else "wrong"
    return "false_code" if pred else "correct_none"


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred: str, truth: str) -> float:
    """1 - edit distance / label length, floored at 0."""
    return max(0.0, 1.0 - edit_distance(pred, truth) / max(1, len(truth)))


def at_level(reads: Iterable[FrameRead], level: str) -> list[FrameRead]:
    """Readings with every status below `level` turned into failures."""
    allowed = LEVELS[level]
    return [r if r.status in allowed else replace(r, status=FAILED, code=None) for r in reads]


@dataclass
class Tally:
    """Outcome counts plus character accuracies for one row of the table."""
    counts: Counter = field(default_factory=Counter)
    char_accs: list[float] = field(default_factory=list)

    def add(self, pred: Optional[str], truth: Optional[str]) -> str:
        o = outcome(pred, truth)
        self.counts[o] += 1
        if pred and truth:
            self.char_accs.append(char_accuracy(pred, truth))
        return o

    @property
    def with_container(self) -> int:
        return self.counts["correct"] + self.counts["wrong"] + self.counts["missed"]

    @property
    def without_container(self) -> int:
        return self.counts["false_code"] + self.counts["correct_none"]

    def rate(self, key: str) -> Optional[float]:
        """Share of container cases (correct/wrong/missed) or of no-container cases."""
        n = self.without_container if key in ("false_code", "correct_none") else self.with_container
        return self.counts[key] / n if n else None

    @property
    def mean_char_accuracy(self) -> Optional[float]:
        return sum(self.char_accs) / len(self.char_accs) if self.char_accs else None


def best_at_level(rotation_reads: Sequence[FrameRead], level: str) -> FrameRead:
    """A frame's best rotation after applying a trust level."""
    return best_reading(at_level(rotation_reads, level))
