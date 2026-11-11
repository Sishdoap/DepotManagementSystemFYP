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

A vehicle's label is a *set* of codes: empty (no container), one, or two
(a 2x20 ft chassis — events 2 and 6 on the footage).

Outcome of one vehicle's predicted codes against its label:

    correct       exactly the labelled codes were output
    partial       some labelled codes were output, none wrong (e.g. only
                  one container of a 2x20)
    wrong         at least one output code is not on the vehicle (the
                  costly error: a wrong container is booked in)
    missed        the vehicle carries a container but no code was output
    false_code    the vehicle carries no container but a code was output
    correct_none  no container and no code

A single frame shows one container, so a frame counts as correct if its
code is any of the vehicle's codes (`outcome`).

Character accuracy compares each output code with the closest labelled
code (1 - edit distance / 11), only where a code was output — a missed
read has no characters to compare.
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
OUTCOMES = ("correct", "partial", "wrong", "missed", "false_code", "correct_none")
CONTAINER_OUTCOMES = ("correct", "partial", "wrong", "missed")


def _as_set(truth) -> frozenset:
    """A label as a set of codes: None/"" -> empty, "A B" or "A" -> {A, B}, iterable -> set."""
    if not truth:
        return frozenset()
    if isinstance(truth, str):
        return frozenset(truth.split())
    return frozenset(truth)


def outcome(pred: Optional[str], truth) -> str:
    """One frame's code against the vehicle's label (any labelled code is correct)."""
    truth = _as_set(truth)
    if truth:
        if not pred:
            return "missed"
        return "correct" if pred in truth else "wrong"
    return "false_code" if pred else "correct_none"


def vehicle_outcome(preds: Sequence[str], truth) -> str:
    """A vehicle's predicted codes against its label (see module docstring)."""
    truth, preds = _as_set(truth), set(preds)
    if not truth:
        return "false_code" if preds else "correct_none"
    if not preds:
        return "missed"
    if preds - truth:
        return "wrong"
    return "correct" if preds == truth else "partial"


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

    def add(self, pred: Optional[str], truth) -> str:
        """Score one frame."""
        o = outcome(pred, truth)
        self.counts[o] += 1
        self._char(pred and [pred], truth)
        return o

    def add_vehicle(self, preds: Sequence[str], truth) -> str:
        """Score one vehicle's fused codes."""
        o = vehicle_outcome(preds, truth)
        self.counts[o] += 1
        self._char(preds, truth)
        return o

    def _char(self, preds, truth) -> None:
        truth = _as_set(truth)
        for p in preds or []:
            if truth:
                self.char_accs.append(max(char_accuracy(p, t) for t in truth))

    @property
    def with_container(self) -> int:
        return sum(self.counts[k] for k in CONTAINER_OUTCOMES)

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
