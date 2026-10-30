"""Fuse per-frame container-code readings into one code per truck (FR1).

A truck passing camera 558 is read on 5-50 frames, at four rotations each
(scripts/read_events.py). Single reads are unreliable on footage: a frame
caught mid-glare reads nothing, a wrong rotation turns door-plate text into
a well-formed but bogus code, and non-container text (fleet numbers,
plates, "INFINITY" on a cab) sometimes passes as an *inferred* code — the
truck scan produced OKMJ5632838 that way. Fusion trades many weak reads
for one strong one.

Two steps:

1. `best_reading`: per frame, keep the strongest rotation, ranked like
   src.iso6346_recovery.reading_key — status, then fewer edits, then
   OCR confidence.
2. A `ReadFuser` combines the frames of one truck (one camera =
   multi-frame; several cameras = cross-camera).

`StatusWeightedVoteFuser` (the default):
    - each frame votes for its code with a weight by status
      (verified 3, recovered 2, inferred 1); failed frames don't vote;
    - a code that was only ever *inferred* must be read on at least
      `min_inferred_support` frames (default 2) — an inferred code is
      unverified, and a single one is as likely to be stray text as a real
      code;
    - the highest-scoring eligible code wins; a tie gives no code
      (status failed, note "tie") rather than a coin flip;
    - the fused status is the strongest status among the frames that read
      the winning code, so "verified" still means at least one frame
      passed the check digit unchanged;
    - size/type (22G1, 45G1, ...) is voted separately by frame count, over
      all frames — it is printed apart from the code and read independently.

Bobtails and box lorries carry no container; for them the right answer is
no code, which the support rule is there to protect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from .iso6346_recovery import FAILED, INFERRED, RECOVERED, STATUS_RANK, VERIFIED


@dataclass(frozen=True)
class FrameRead:
    """The reading of one frame (at one rotation)."""
    camera: str
    frame: int
    status: str                      # verified | recovered | inferred | failed
    code: Optional[str] = None
    size_type: Optional[str] = None
    edits: int = 0
    conf: float = 0.0
    rotation: int = 0


def best_reading(reads: Iterable[FrameRead]) -> FrameRead:
    """The strongest of several readings of one frame (e.g. one per rotation).

    Ranked by status, then fewer edits, then confidence; on a full tie the
    first wins, so rotation order decides deterministically.
    """
    reads = list(reads)
    if not reads:
        raise ValueError("no readings")
    return max(reads, key=lambda r: (STATUS_RANK[r.status], -r.edits, r.conf))


@dataclass
class FusedRead:
    """One truck's fused code."""
    code: Optional[str]
    status: str                      # strongest status among supporting frames, or failed
    support: int = 0                 # frames that read `code`
    frames: int = 0                  # frames considered
    cameras: tuple[str, ...] = ()    # cameras with at least one supporting frame
    size_type: Optional[str] = None
    size_type_support: int = 0
    scores: dict[str, float] = field(default_factory=dict)   # every voted code -> score
    note: str = ""


class ReadFuser(ABC):
    """Combines the frame readings of one truck into one code."""

    @abstractmethod
    def fuse(self, reads: Sequence[FrameRead]) -> FusedRead:
        """Fuse readings, at most one per (camera, frame) — see best_reading."""

    @property
    def name(self) -> str:
        return type(self).__name__


DEFAULT_WEIGHTS = {VERIFIED: 3.0, RECOVERED: 2.0, INFERRED: 1.0}


class StatusWeightedVoteFuser(ReadFuser):
    """Status-weighted vote; inferred-only codes need several frames (module docstring)."""

    def __init__(self, weights: Optional[Mapping[str, float]] = None,
                 min_inferred_support: int = 2):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.min_inferred_support = min_inferred_support

    def fuse(self, reads: Sequence[FrameRead]) -> FusedRead:
        score: dict[str, float] = defaultdict(float)
        support: Counter = Counter()
        strongest: dict[str, str] = {}
        cams: dict[str, set] = defaultdict(set)
        for r in reads:
            if r.status == FAILED or not r.code:
                continue
            score[r.code] += self.weights.get(r.status, 0.0)
            support[r.code] += 1
            cams[r.code].add(r.camera)
            if STATUS_RANK[r.status] > STATUS_RANK.get(strongest.get(r.code, FAILED), 0):
                strongest[r.code] = r.status

        size_type, size_support = _vote_size_type(reads)
        base = dict(frames=len(reads), size_type=size_type, size_type_support=size_support,
                    scores=dict(sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))))

        eligible = [c for c in score
                    if strongest[c] != INFERRED or support[c] >= self.min_inferred_support]
        if not eligible:
            note = ""
            if score:
                top = max(score, key=lambda c: (score[c], c))
                note = f"rejected: {top} inferred on {support[top]} frame(s)"
            return FusedRead(None, FAILED, note=note, **base)

        ranked = sorted(eligible, key=lambda c: (-score[c], c))
        if len(ranked) > 1 and score[ranked[0]] == score[ranked[1]]:
            return FusedRead(None, FAILED, note=f"tie: {ranked[0]} vs {ranked[1]}", **base)
        winner = ranked[0]
        note = f"also read: {', '.join(ranked[1:])}" if len(ranked) > 1 else ""
        return FusedRead(winner, strongest[winner], support=support[winner],
                         cameras=tuple(sorted(cams[winner])), note=note, **base)


def _vote_size_type(reads: Sequence[FrameRead]) -> tuple[Optional[str], int]:
    """Most frequent size/type over all frames; None on a tie or if never read."""
    counts = Counter(r.size_type for r in reads if r.size_type)
    if not counts:
        return None, 0
    (top, n), *rest = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if rest and rest[0][1] == n:
        return None, 0
    return top, n
