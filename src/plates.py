"""Malaysian licence plates from OCR fragments, fused per truck (FR4).

Baseline without a trained plate detector: scripts/read_plates.py runs
PaddleOCR on the truck crop of every frame and stores each text fragment
with its position; this module turns a frame's fragments into plate
candidates and fuses them per truck.

What the footage shows (2026-11-02):

- Two plates per container truck. The **vehicle plate** belongs to the
  prime mover: on the front bumper (camera 561, side cameras on arrival)
  and repeated small at the rear. The **trailer plate** is on the trailer
  rear, prefixed "T/" (e.g. "T/BF 7421"), read by 558 and the side
  cameras as the truck leaves. Box lorries and light vehicles have only a
  vehicle plate. So plates are modelled as vehicle / trailer, not
  front / rear.
- Plates come as one line ("BQB 7404") or two lines ("T/BF" above
  "7421"); OCR returns a two-line plate as two fragments, joined here when
  a letters fragment sits directly above a digits fragment.
- Fleet numbers painted on the cab (INF403, SG328) have the same shape as
  a plate. They are high on the vehicle, plates are low, so fragments in
  the top part of the truck box (`min_y` of the box height) are ignored.
  On the side cameras some fleet numbers are painted low on the cab door
  (XPS310, SG328 read on 6-15 frames); they lost every vote on the
  footage, but this is the known weak spot of the baseline.

Plate format (Peninsular Malaysia, also covers most commercial plates):
    vehicle   1-3 letters, 1-4 digits, optional 1 letter   e.g. WDX7232, BQB7404
    trailer   "T/" + the same                               e.g. T/BF7421
Accepted here: at least MIN_LETTERS letters and MIN_DIGITS digits (see
the constants for why).
Look-alike characters are fixed by position (digit part O->0, I->1, ...;
letter part 0->Q, 8->B, ...) — the same type-forced idea as the container
code reader — but at most one fix per plate (MAX_FIXES), and the letter
part must contain a real letter. Malaysian plates do not use I or O, so a
letter part that still contains them is rejected (which also drops fleet
numbers like INF403).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

VEHICLE = "vehicle"
TRAILER = "trailer"

# More than one fix turns unrelated text into plates ("22G1" -> "ZZG1").
MAX_FIXES = 1
# Commercial plates on the footage all have 2-3 letters and 4 digits. The
# format allows 1-letter and 1-digit plates (private vanity plates), but
# accepting them lets short words through: "CUBE" from a container's
# capacity plate reads as CU8E, partial reads give "BQ8".
MIN_LETTERS = 2
MIN_DIGITS = 3
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
_TO_LETTER = {"0": "Q", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A"}
# letters (no I/O) + digits + optional suffix letter
_PLATE_RE = re.compile(r"^([A-HJ-NP-Z]{1,3})([0-9]{1,4})([A-HJ-NP-Z]?)$")
_LETTERS_RE = re.compile(r"^(T/)?[A-Z0-9]{1,3}$")
_DIGITS_RE = re.compile(r"^[0-9OIDQLZSBG]{1,4}[A-Z]?$")


@dataclass(frozen=True)
class Fragment:
    """One OCR text fragment, box in fractions of the truck crop (0..1)."""
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class PlateRead:
    """A plate candidate read on one frame."""
    plate: str          # normalised, e.g. "BQB7404" or "T/BF7421"
    kind: str           # VEHICLE or TRAILER
    conf: float
    camera: str = ""
    frame: int = 0
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # in the truck box, 0..1


def normalise_plate(text: str) -> Optional[tuple[str, str]]:
    """(plate, kind) if `text` is a well-formed plate after look-alike fixes, else None.

    Every split into letters (1-3) + digits (1-4) + optional suffix letter
    is tried; the one needing the fewest look-alike fixes wins, ties going
    to more letters and no suffix — so "SG328" stays SG328 rather than
    becoming S6328, and "8QB7404" becomes BQB7404.
    """
    t = re.sub(r"[^A-Z0-9/]", "", text.upper())
    kind = VEHICLE
    if t.startswith("T/"):
        kind, t = TRAILER, t[2:]
    t = t.replace("/", "")
    best = None                                  # (fixes, -letters, has_suffix, plate)
    for n in range(3, MIN_LETTERS - 1, -1):
        letters_raw, rest = t[:n], t[n:]
        options = [(rest, "")]
        if len(rest) >= 2 and rest[-1].isalpha():
            options.append((rest[:-1], rest[-1]))
        for digits_raw, suffix in options:
            if len(digits_raw) < MIN_DIGITS:
                continue
            letters = "".join(_TO_LETTER.get(c, c) for c in letters_raw)
            digits = "".join(_TO_DIGIT.get(c, c) for c in digits_raw)
            plate = letters + digits + suffix
            if not _PLATE_RE.match(plate) or not any(c.isalpha() for c in letters_raw):
                continue
            fixes = sum(a != b for a, b in zip(letters_raw + digits_raw, letters + digits))
            if fixes > MAX_FIXES:
                continue
            key = (fixes, -n, bool(suffix), plate)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    return (("T/" if kind == TRAILER else "") + best[3], kind)


def _stack_two_line(frags: list[Fragment]) -> list[str]:
    """Join a letters fragment directly above a digits fragment (two-line plates)."""
    joined = []
    for top in frags:
        if not _LETTERS_RE.match(re.sub(r"[^A-Z0-9/]", "", top.text.upper())):
            continue
        h = top.y2 - top.y1
        for bottom in frags:
            if bottom is top or not _DIGITS_RE.match(re.sub(r"[^A-Z0-9]", "", bottom.text.upper())):
                continue
            below = 0 <= bottom.cy - top.cy <= 2.5 * h
            overlap = min(top.x2, bottom.x2) - max(top.x1, bottom.x1)
            if below and overlap > 0.5 * min(top.x2 - top.x1, bottom.x2 - bottom.x1):
                joined.append(top.text + bottom.text)
    return joined


def plates_in_frame(frags: Sequence[Fragment], *, min_y: float = 0.4,
                    camera: str = "", frame: int = 0) -> list[PlateRead]:
    """Plate candidates among one frame's fragments.

    Fragments whose centre is above `min_y` (fraction of the truck box
    height from the top) are ignored: that is where fleet numbers and
    company names are painted.
    """
    low = [f for f in frags if f.cy >= min_y and f.text]
    out, seen = [], set()
    candidates = [(f.text, f.conf, (f.x1, f.y1, f.x2, f.y2)) for f in low]
    candidates += [(t, min(a.conf, b.conf),
                    (min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2)))
                   for t, a, b in _pairs(low)]
    for text, conf, box in candidates:
        parsed = normalise_plate(text)
        if parsed and parsed[0] not in seen:
            seen.add(parsed[0])
            out.append(PlateRead(parsed[0], parsed[1], conf, camera, frame, box))
    return out


def _pairs(frags: list[Fragment]):
    """(joined text, top, bottom) for two-line plates."""
    for top in frags:
        for bottom in frags:
            if top is bottom:
                continue
            for t in _stack_two_line([top, bottom]):
                yield t, top, bottom


@dataclass
class FusedPlate:
    plate: Optional[str]
    support: int = 0                 # frames that read `plate`
    frames: int = 0                  # frames with any candidate of this kind
    cameras: tuple[str, ...] = ()
    note: str = ""
    evidence: Optional[PlateRead] = None   # the most confident read of `plate` (for crops)


def fuse_plates(reads: Iterable[PlateRead], kind: str, *, min_support: int = 2) -> FusedPlate:
    """Vote one plate of `kind` per truck: most frames wins, ties give none.

    Each frame votes once per distinct plate. A plate read on fewer than
    `min_support` frames is rejected — a single read is as likely to be
    stray text as a plate, the lesson from container codes.

    For the vehicle plate, candidates that are a trailer plate with its
    "T/" misread (slash dropped: "TBF7421"; T read as U: "UBF6048") are
    ignored: a vehicle candidate whose text after the first character
    equals the body of a trailer plate read on the same truck.
    """
    reads = list(reads)
    trailer_bodies = {r.plate[2:] for r in reads if r.kind == TRAILER}
    frames_by_plate: dict[str, set] = defaultdict(set)
    cams: dict[str, set] = defaultdict(set)
    best: dict[str, PlateRead] = {}
    all_frames = set()
    for r in reads:
        if r.kind != kind:
            continue
        if kind == VEHICLE and r.plate[1:] in trailer_bodies:
            continue
        frames_by_plate[r.plate].add((r.camera, r.frame))
        cams[r.plate].add(r.camera)
        all_frames.add((r.camera, r.frame))
        if r.plate not in best or r.conf > best[r.plate].conf:
            best[r.plate] = r
    if not frames_by_plate:
        return FusedPlate(None)
    ranked = sorted(frames_by_plate, key=lambda p: (-len(frames_by_plate[p]), p))
    top = ranked[0]
    n = len(frames_by_plate[top])
    if n < min_support:
        return FusedPlate(None, frames=len(all_frames), note=f"rejected: {top} on {n} frame(s)")
    if len(ranked) > 1 and len(frames_by_plate[ranked[1]]) == n:
        return FusedPlate(None, frames=len(all_frames), note=f"tie: {top} vs {ranked[1]}")
    others = [p for p in ranked[1:] if len(frames_by_plate[p]) >= min_support]
    return FusedPlate(top, n, len(all_frames), tuple(sorted(cams[top])),
                      note=f"also read: {', '.join(others)}" if others else "",
                      evidence=best[top])


def load_plate_fragments(path) -> dict[tuple[int, str, int], list[Fragment]]:
    """Read scripts/read_plates.py output: {(event, camera, frame): [Fragment, ...]}."""
    import csv

    frames: dict[tuple[int, str, int], list[Fragment]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            frames[(int(r["event"]), r["camera"], int(r["frame"]))].append(Fragment(
                r["text"], float(r["conf"]), *(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    return dict(frames)
