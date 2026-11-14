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
    - every code must be read on at least `min_support` frames (default 2),
      whatever its status. A single verified or recovered frame used to be
      enough; on 13 Nov 11:00 the only three such codes (BICU1202510,
      ZZEZ2010810, RRHU1750100 — check digit passed by chance) were all on
      vehicles without a container (a bare chassis, two prime movers
      without a trailer). On the 12 Aug hour (development set) the
      all-camera result is unchanged at 2 (10/10, no false code); single
      cameras lose two correct single-frame reads and one wrong one;
    - a code that was only ever *inferred* must also reach
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

`MultiContainerFuser` adds a second code on a 2x20 ft truck, and
`resolve_shared_codes` then runs across all trucks so that a container
caught from a neighbouring truck is not counted on two at once.
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
    box_px: float = 0.0              # longest side of the truck box (frame size proxy)


def load_frame_reads(path) -> dict[tuple[int, str, int], list[FrameRead]]:
    """Read scripts/read_events.py output: {(event, camera, frame): [one FrameRead per rotation]}."""
    import csv

    by_frame: dict[tuple[int, str, int], list[FrameRead]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_frame[(int(r["event"]), r["camera"], int(r["frame"]))].append(FrameRead(
                camera=r["camera"], frame=int(r["frame"]), status=r["status"],
                code=r["code"] or None, size_type=r["size_type"] or None,
                edits=int(r["edits"]), conf=float(r["mean_conf"]), rotation=int(r["rotation"]),
                box_px=float(r["box_px"]),
            ))
    return dict(by_frame)


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
    codes: tuple[str, ...] = ()      # every container on the truck, `code` first

    def __post_init__(self):
        if self.code and not self.codes:
            self.codes = (self.code,)


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
                 min_inferred_support: int = 2, min_support: int = 2):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.min_inferred_support = min_inferred_support
        self.min_support = min_support

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

        eligible = [c for c in score if support[c] >= self.min_support
                    and (strongest[c] != INFERRED or support[c] >= self.min_inferred_support)]
        if not eligible:
            note = ""
            if score:
                top = max(score, key=lambda c: (score[c], c))
                note = f"rejected: {top} {strongest[top]} on {support[top]} frame(s)"
            return FusedRead(None, FAILED, note=note, **base)

        ranked = sorted(eligible, key=lambda c: (-score[c], c))
        # A tie gives no code - counting codes too weak to be reported
        # themselves: a rejected rival with as much evidence still means the
        # reads disagree (13 Nov 11:00 event 16, a prime mover without a
        # container: GMVJ6224557 inferred on 2 frames vs WTHU0470422 recovered
        # on 1, score 2 each).
        rival = max((c for c in score if c != ranked[0]), key=lambda c: (score[c], c), default=None)
        if rival is not None and score[rival] >= score[ranked[0]]:
            return FusedRead(None, FAILED, note=f"tie: {ranked[0]} vs {rival}", **base)
        winner = ranked[0]
        note = f"also read: {', '.join(ranked[1:])}" if len(ranked) > 1 else ""
        return FusedRead(winner, strongest[winner], support=support[winner],
                         cameras=tuple(sorted(cams[winner])), note=note, **base)


class MultiContainerFuser(ReadFuser):
    """One or two containers per truck: a 2x20 ft chassis carries two codes.

    On the footage two of ten container trucks (events 2 and 6) carried two
    20 ft containers. The rear camera (558) sees only the back container's
    doors; the side cameras (557/560) see both, and read the front one's
    code on many frames. A single-winner vote reports only one of them.

    Rule: the primary code comes from `base` (StatusWeightedVoteFuser by
    default). A second code is added if
      - it was read *validated* (verified or recovered) on at least
        `min_extra_support` frames — a stray validated misread happens on
        one or two frames (event 15: CRSU2240537 on 2 frames, a misread of
        CRSU1138578), a real second container on many (8 and 25);
      - the voted size/type is a 20 ft code (first character "2") — a
        40/45 ft container leaves no room for another on the chassis;
      - the second code's *own* size/type, voted over the frames that
        read it, is not a 40/45 ft code either (13 Nov 11:00, event 10:
        the side cameras caught the 45G1 container of the truck 15 s
        ahead, TXGU4947095, on 11 frames, and it was added to a 22G1
        truck). A second code whose size was never read is still allowed;
      - at most `max_codes` codes in total (two 20 ft boxes per chassis).
    The threshold of 4 sits between those observations; with only two
    real examples it is a heuristic to re-check on more footage.

    Fusion sees one truck at a time; a container that leaks in from a
    neighbouring truck with the same size is only caught across trucks,
    by `resolve_shared_codes`.
    """

    def __init__(self, base: Optional[ReadFuser] = None, min_extra_support: int = 4,
                 max_codes: int = 2):
        self.base = base or StatusWeightedVoteFuser()
        self.min_extra_support = min_extra_support
        self.max_codes = max_codes

    def fuse(self, reads: Sequence[FrameRead]) -> FusedRead:
        out = self.base.fuse(reads)
        if not out.code or not (out.size_type or "").startswith("2"):
            return out
        validated = Counter(r.code for r in reads
                            if r.code and r.code != out.code
                            and STATUS_RANK[r.status] >= STATUS_RANK[RECOVERED])
        candidates = [c for c, n in sorted(validated.items(), key=lambda kv: (-kv[1], kv[0]))
                      if n >= self.min_extra_support]
        notes, extra = [], []
        for c in candidates:
            own_size, _ = _vote_size_type([r for r in reads if r.code == c])
            if own_size and not own_size.startswith("2"):
                notes.append(f"not a second container: {c} reads as {own_size}")
            elif len(extra) < self.max_codes - 1:
                extra.append(c)
        if extra:
            out.codes = (out.code, *extra)
            notes.insert(0, "second container: "
                         + ", ".join(f"{c} ({validated[c]} frames)" for c in extra))
        if notes:
            out.note = "; ".join(notes + ([out.note] if out.note else []))
        return out


def resolve_shared_codes(fused: Mapping[int, FusedRead], times: Mapping[int, "datetime"],
                         window_s: float = 180.0) -> None:
    """Give each container to one truck when trucks close in time share it (in place).

    A container is on one truck at a time. When trucks pass seconds apart,
    a side camera can catch the neighbour's container and attach it to
    the wrong event as a second container (13 Nov 11:00: event 10 listed
    event 9's TXGU4947095). Fusion sees one truck at a time, so this runs
    over all fused events afterwards.

    For every code listed by two or more events whose entry times are
    within `window_s`:
      - if exactly one of them has it as its main code, it is removed from
        the others' second containers;
      - if it is only a second container, the event with the higher vote
        score for it keeps it; on a tie, none does;
      - if several have it as the main code (the same truck split in two,
        or a real misassignment) those main codes stay, each with a note
        naming the other events for a person to check; it is still
        removed from any other event's second containers.
    The window keeps a container that legitimately passes twice — in at
    the gate, out hours later — on both trucks.
    """
    from datetime import timedelta

    window = timedelta(seconds=window_s)
    holders: dict[str, list[int]] = defaultdict(list)
    for ev, out in fused.items():
        for c in out.codes:
            holders[c].append(ev)

    def drop(ev: int, code: str, why: str) -> None:
        out = fused[ev]
        out.codes = tuple(c for c in out.codes if c != code)
        note = f"dropped second container {code} ({why})"
        out.note = f"{note}; {out.note}" if out.note else note

    for code, evs in sorted(holders.items()):
        for group in _close_groups(sorted(evs, key=lambda e: times[e]), times, window):
            if len(group) < 2:
                continue
            owners = [e for e in group if fused[e].code == code]
            if len(owners) == 1:
                for e in group:
                    if e != owners[0]:
                        drop(e, code, f"main code of event {owners[0]}")
            elif not owners:
                best = max(fused[e].scores.get(code, 0.0) for e in group)
                keep = [e for e in group if fused[e].scores.get(code, 0.0) == best]
                for e in group:
                    if len(keep) > 1 or e != keep[0]:
                        why = (f"tie with events {', '.join(map(str, keep))}" if len(keep) > 1
                               else f"more support on event {keep[0]}")
                        drop(e, code, why)
            else:
                for e in group:
                    if e not in owners:
                        drop(e, code, f"main code of events {', '.join(map(str, owners))}")
                for e in owners:
                    others = ", ".join(str(o) for o in owners if o != e)
                    note = f"{code} is also the code of event {others}"
                    if note in fused[e].note:      # live mode re-runs this as trucks arrive
                        continue
                    fused[e].note = f"{note}; {fused[e].note}" if fused[e].note else note


def _close_groups(evs: Sequence[int], times: Mapping[int, "datetime"], window) -> list[list[int]]:
    """Split time-sorted events into runs whose consecutive gaps are within `window`."""
    groups: list[list[int]] = []
    for e in evs:
        if groups and times[e] - times[groups[-1][-1]] <= window:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def _vote_size_type(reads: Sequence[FrameRead]) -> tuple[Optional[str], int]:
    """Most frequent size/type over all frames; None on a tie or if never read."""
    counts = Counter(r.size_type for r in reads if r.size_type)
    if not counts:
        return None, 0
    (top, n), *rest = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if rest and rest[0][1] == n:
        return None, 0
    return top, n
