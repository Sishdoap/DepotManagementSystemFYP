"""Reading ISO 6346 container codes from noisy OCR fragments.

Used by RealOCRAdapter. PaddleOCR returns a list of text fragments, e.g.
["TEMU", "6372145"] — or ["6372145", "TEMU"] when the crop is rotated, or
["CSNU45G1740176"] when the size/type code gets glued on. This module turns
those fragments into a container code *and* a statement of how far to
trust it.

Pipeline (`read_container_code`):

  1. Clean the fragments, drop weight-plate text, and pull out ISO
     size/type codes (45G1, 22G1, and garbled ones like "4561") so they
     don't contaminate the container code. The first one found is returned
     as `size_type` (groundwork for load-profile recognition).
  2. Build candidate strings: the code assembled from fragment roles
     (owner + serial + boxed check digit, whatever order OCR returned them
     in), the text in reading order with windows at each owner code, and —
     for a single glued fragment — the owner code moved to the front.
  3. For each candidate, look for a valid code with the fewest changes:
       - type-forced fixes are always applied (a digit where a letter must
         be or vice versa: 0->O, O->0, 1->I, ...);
       - optionally, up to `max_confusions` look-alike substitutions
         (off by default — they add more wrong codes than right ones);
       - candidates are tried in a fixed order, so the result never depends
         on Python's hash seed;
       - if two different valid codes need the same number of changes, or
         two arrangements give different validated codes, the read is
         rejected as ambiguous rather than guessed.
  4. If nothing verifies but the first ten characters are well formed, the
     check digit is computed from them: status "inferred". That is the
     usual OCR failure — the check digit is printed inside a box that the
     recogniser misreads or drops — but nothing then verifies the other ten
     characters, so inferred codes are reported separately and never
     counted as validated.

Statuses, strongest first:
    verified   the raw read passed the check digit unchanged
    recovered  valid after the fixes above; the check digit was not touched,
               so it still independently confirms the result
    inferred   check digit missing or wrong and recomputed — NOT verified
    failed     nothing usable, or ambiguous

Every substitution or arrangement tried is another chance for a wrong code
to pass the check digit by luck (roughly 1 in 10 each), so `ReaderConfig`
exposes the options and scripts/replay_recovery.py measures the trade-off on
the labelled images. On 1,386 images the defaults validate 74.8% of codes
correctly with 9 validated-but-wrong (FYP1: 67.7% and 26), and infer another
18% (80% of inferred codes correct).

This replaces the earlier port of predict_paddle.py's correct_to_valid_iso6346,
which iterated Python sets, returned the first valid hit rather than the
smallest fix, ignored the U/J/Z category rule and only accepted exactly 11
characters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Iterable, Mapping, Optional

from .iso6346 import compute_check_digit, validate


VERIFIED = "verified"
RECOVERED = "recovered"
INFERRED = "inferred"
FAILED = "failed"
STATUS_RANK = {VERIFIED: 3, RECOVERED: 2, INFERRED: 1, FAILED: 0}

CATEGORIES = "UJZ"

# Type-forced conversions: applied whenever a character of the wrong type
# sits in a position, because the position's type is certain.
LETTER_FROM_DIGIT = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S",
                     "6": "G", "7": "T", "8": "B"}
DIGIT_FROM_LETTER = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                     "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7",
                     "B": "8"}


def symmetric_pairs(pairs: Iterable[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    """Build a confusion table {char: alternatives} from unordered pairs."""
    out: dict[str, set[str]] = {}
    for a, b in pairs:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return {k: tuple(sorted(v)) for k, v in out.items()}


# Same-type confusion candidates (each substitution costs one edit): pairs
# observed in the FYP1 benchmark or common in stencilled fonts.
DIGIT_CONFUSIONS = symmetric_pairs([
    ("0", "8"), ("1", "7"), ("1", "4"), ("3", "8"), ("3", "9"),
    ("5", "6"), ("6", "8"), ("8", "9"),
])
LETTER_CONFUSIONS = symmetric_pairs([
    ("O", "Q"), ("O", "D"), ("D", "Q"), ("I", "L"), ("I", "T"),
    ("C", "G"), ("E", "F"), ("M", "N"), ("N", "Z"), ("P", "R"),
    ("U", "V"),
])

# Common ISO 6346 size/type codes: length (1/2/4/L) + height (0/2/5) +
# type group + detail digit, e.g. 22G1, 42U1, 45G1, 45R1, L5G1.
SIZE_TYPE_RE = re.compile(r"[124L][025][GRUTPBHS][0-9]")
# A whole fragment that is a size/type code as OCR typically garbles it:
# "45G", "4561" (G read as 6), "22C1", "2ZG1", "5G1". Dropped so it can't be
# glued onto the container code.
SIZE_TYPE_FRAGMENT_RE = re.compile(r"^(?:[124L][025Z][GC6RU][01I]?|[025Z][GC6R][01I])$")
# Words from the weight/capacity plate that sometimes land in the crop.
PLATE_TEXT_RE = re.compile(r"KGS?|LBS?|CUM|CUFT|TARE|GROSS|NET|PAYLOAD|MAX")


@dataclass(frozen=True)
class ReaderConfig:
    """Knobs for `read_container_code` (see module docstring).

    Defaults are measured by scripts/replay_recovery.py on 1,386 labelled
    images (results/recovery_ablation.md): look-alike substitutions and the
    check-digit-first rearrangement each added more wrong-but-validated
    codes than correct ones, so both are off; only type-forced fixes stay.
    """
    max_confusions: int = 0
    digit_confusions: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: DIGIT_CONFUSIONS)
    letter_confusions: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: LETTER_CONFUSIONS)
    allow_infer: bool = True
    # Also try owner + serial + check when the check digit was read first
    # (e.g. "CSLU7236936" for CSLU 236936 7).
    try_check_digit_first: bool = False
    # If different candidate arrangements yield different equally strong
    # codes, report the read as ambiguous instead of picking one.
    reject_conflicts: bool = True


DEFAULT_CONFIG = ReaderConfig()


@dataclass(frozen=True)
class CodeReading:
    """Outcome of reading a container code from OCR output."""
    code: Optional[str]          # best code (also set for inferred), else None
    status: str                  # verified | recovered | inferred | failed
    raw: str                     # the candidate string the code was read from
    edits: int                   # characters changed from `raw`
    size_type: Optional[str] = None
    note: str = ""               # e.g. "ambiguous: 2 codes at minimal edits"

    @property
    def is_validated(self) -> bool:
        """True if the check digit confirms the code (verified or recovered)."""
        return self.status in (VERIFIED, RECOVERED)


def is_valid_container_code(code: str) -> bool:
    """Strict check: exactly 11 upper-case characters passing ISO 6346."""
    return len(code) == 11 and code == code.upper() and validate(code).is_valid


def clean(text: str) -> str:
    """Upper-case and keep only A-Z / 0-9."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# --- Single candidate string ---

def _position_options(pos: int, ch: str, cfg: ReaderConfig) -> tuple[Optional[str], tuple[str, ...]]:
    """(zero-cost character, one-edit alternatives) for position `pos`.

    The zero-cost character is the read character, or its type-forced
    conversion; None if the position can only be fixed by a substitution.
    """
    if pos <= 3:
        base = ch if ch.isalpha() else LETTER_FROM_DIGIT.get(ch)
        alts = cfg.letter_confusions.get(base, ()) if base else ()
        if pos == 3:
            alts = tuple(c for c in alts if c in CATEGORIES)
            if base is None or base not in CATEGORIES:
                base = None
        return base, alts
    base = ch if ch.isdigit() else DIGIT_FROM_LETTER.get(ch)
    if pos == 10:
        return base, ()     # changing the check digit would verify nothing
    return base, (cfg.digit_confusions.get(base, ()) if base else ())


def _search_verified(s: str, cfg: ReaderConfig) -> list[str]:
    """Valid codes reachable from 11-char `s` with the fewest confusion edits.

    Returns the sorted codes found at the smallest number of substitutions
    that yields any valid code, or [] if none within `cfg.max_confusions`.
    """
    options = [_position_options(i, c, cfg) for i, c in enumerate(s)]
    if any(base is None and not alts for base, alts in options):
        return []
    mandatory = {i for i, (base, _) in enumerate(options) if base is None}
    swappable = [i for i, (_, alts) in enumerate(options) if alts]

    for k in range(len(mandatory), cfg.max_confusions + 1):
        found: set[str] = set()
        for chosen in combinations(swappable, k):
            if not mandatory <= set(chosen):
                continue
            choices = [options[i][1] if i in chosen else (options[i][0],) for i in range(11)]
            for combo in product(*choices):
                code = "".join(combo)
                if is_valid_container_code(code):
                    found.add(code)
        if found:
            return sorted(found)
    return []


def _infer(s10: str, cfg: ReaderConfig) -> Optional[str]:
    """Append a computed check digit to ten characters, after forced fixes."""
    fixed = []
    for i, ch in enumerate(s10):
        base, _ = _position_options(i, ch, cfg)
        if base is None:
            return None
        fixed.append(base)
    prefix = "".join(fixed)
    return prefix + str(compute_check_digit(prefix))


def _diff(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def read_candidate(s: str, cfg: ReaderConfig = DEFAULT_CONFIG) -> CodeReading:
    """Read one already-cleaned candidate string (10 or 11 characters).

    There is deliberately no "delete one character" handling for 12-character
    reads: trying all 12 deletions gives a wrong code about as often as a
    right one (measured on the FYP1 image set).
    """
    if len(s) == 11:
        found = _search_verified(s, cfg)
        if len(found) == 1:
            code = found[0]
            edits = _diff(code, s)
            return CodeReading(code, VERIFIED if edits == 0 else RECOVERED, s, edits)
        if len(found) > 1:
            return CodeReading(None, FAILED, s, 0,
                               note=f"ambiguous: {len(found)} codes at minimal edits")
        if cfg.allow_infer:
            code = _infer(s[:10], cfg)
            if code:
                return CodeReading(code, INFERRED, s, _diff(code, s),
                                   note="check digit wrong; recomputed")
        return CodeReading(None, FAILED, s, 0)

    if len(s) == 10:
        code = _infer(s, cfg) if cfg.allow_infer else None
        if code:
            return CodeReading(code, INFERRED, s, _diff(code, s),
                               note="check digit missing; computed")
        return CodeReading(None, FAILED, s, 0)

    return CodeReading(None, FAILED, s, 0)


# --- Fragments ---

def _normalise_size_type(frag: str) -> Optional[str]:
    """Undo common OCR garbling of a 4-char size/type fragment ("4561" -> "45G1")."""
    if len(frag) != 4:
        return None
    fixed = frag[0] + frag[1].replace("Z", "2") + frag[2].replace("6", "G").replace("C", "G") \
        + frag[3].replace("I", "1")
    return fixed if SIZE_TYPE_RE.fullmatch(fixed) else None


def extract_size_type(fragments: Iterable[str]) -> tuple[list[str], Optional[str]]:
    """Remove size/type codes and plate text from cleaned fragments.

    Returns the remaining fragments and the first size/type code found.
    Glued codes ("CSNU45G1740176") are cut out of the fragment; whole
    fragments that look like a garbled size/type code ("45G", "4561") or
    weight-plate text ("2OKGS") are dropped.
    """
    size_type = None
    out = []
    for frag in fragments:
        if PLATE_TEXT_RE.search(frag):
            continue
        if SIZE_TYPE_FRAGMENT_RE.fullmatch(frag) and not SIZE_TYPE_RE.fullmatch(frag):
            size_type = size_type or _normalise_size_type(frag)
            continue
        m = SIZE_TYPE_RE.search(frag)
        while m:
            size_type = size_type or m.group()
            frag = frag[:m.start()] + frag[m.end():]
            m = SIZE_TYPE_RE.search(frag)
        if frag:
            out.append(frag)
    return out, size_type


def _is_digitish(ch: str) -> bool:
    return ch.isdigit() or ch in DIGIT_FROM_LETTER


def _owner_positions(text: str) -> list[int]:
    """Start indices of plausible owner+category codes (4 chars ending U/J/Z)."""
    hits = []
    for i in range(len(text) - 3):
        head, cat = text[i:i + 3], text[i + 3]
        if cat in CATEGORIES and any(c.isalpha() for c in head) and \
                all(c.isalpha() or c in LETTER_FROM_DIGIT for c in head):
            hits.append(i)
    return hits


def _digit_run(text: str, start: int) -> int:
    """Length of the run of digit-like characters starting at `start`."""
    n = 0
    while start + n < len(text) and _is_digitish(text[start + n]):
        n += 1
    return n


def _assembled_by_role(fragments: list[str]) -> list[str]:
    """Assemble a code from fragment roles, ignoring their reading order.

    The check digit sits in a box, so OCR often returns it as its own
    fragment — and at 180 degrees the fragment order is reversed
    ("4 | 670973 | CSNU" for CSNU 670973 4). With exactly one owner fragment
    the code is owner + 7-digit fragment, or owner + 6-digit serial +
    1-digit check, or owner + serial alone (check digit missing).
    """
    owners = [i for i, f in enumerate(fragments) if len(f) == 4 and _owner_positions(f) == [0]]
    if len(owners) != 1:
        return []
    owner = fragments[owners[0]]
    digits = [f for i, f in enumerate(fragments)
              if i != owners[0] and all(_is_digitish(c) for c in f)]
    by_len = {n: [f for f in digits if len(f) == n] for n in (1, 6, 7)}
    out = []
    if len(by_len[7]) == 1:
        out.append(owner + by_len[7][0])
    if len(by_len[6]) == 1 and len(by_len[1]) == 1:
        out.append(owner + by_len[6][0] + by_len[1][0])
    if len(by_len[6]) == 1 and not by_len[1] and not by_len[7]:
        out.append(owner + by_len[6][0])
    return out


def candidate_strings(
    fragments: list[str], cfg: ReaderConfig = DEFAULT_CONFIG,
) -> list[tuple[str, bool]]:
    """Candidate code strings in priority order, without duplicates.

    Returns (candidate, speculative) pairs:
      1. codes assembled from fragment roles (see `_assembled_by_role`);
      2. the text in reading order, and windows starting at each owner code;
      3. for a single fragment with digits before the owner code
         ("7773935CCLU"), the owner moved to the front;
      4. speculative: the check digit moved from first to last. These only
         count if they verify exactly, so one guess is never stacked on
         another.
    Every extra arrangement is another chance for a wrong code to pass by
    luck, so arrangements are limited to these observed OCR patterns.
    """
    out: list[tuple[str, bool]] = [(c, False) for c in _assembled_by_role(fragments)]
    joined = "".join(fragments)
    texts = [joined]
    owners = _owner_positions(joined)
    if len(fragments) == 1 and owners and owners[0] > 0:
        p = owners[0]
        texts.append(joined[p:p + 4] + joined[p + 4:] + joined[:p])

    for text in texts:
        if 10 <= len(text) <= 11:
            out.append((text, False))
        for p in _owner_positions(text):
            run = _digit_run(text, p + 4)
            if run >= 7:
                out.append((text[p:p + 11], False))
            if run == 6:
                out.append((text[p:p + 10], False))    # check digit genuinely missing
            if run >= 7 and cfg.try_check_digit_first:
                out.append((text[p:p + 4] + text[p + 5:p + 11] + text[p + 4], True))
    seen: set[str] = set()
    return [c for c in out if not (c[0] in seen or seen.add(c[0]))]


def reading_key(reading: CodeReading, fragments: Iterable[tuple[str, float]]) -> tuple:
    """Sort key for choosing between readings (e.g. across rotations).

    Stronger status first, then fewer edits, then higher mean OCR confidence.
    """
    confs = [c for _, c in fragments]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return (STATUS_RANK[reading.status], -reading.edits, mean_conf)


def read_container_code(
    fragments: Iterable[tuple[str, float]] | Iterable[str],
    cfg: ReaderConfig = DEFAULT_CONFIG,
) -> CodeReading:
    """Read the best container code from OCR fragments.

    Args:
        fragments: (text, confidence) pairs or plain strings, in reading order.
        cfg: reader configuration; the defaults are measured, not guessed.
    """
    texts = [f[0] if isinstance(f, (tuple, list)) else f for f in fragments]
    cleaned = [c for c in (clean(t) for t in texts) if c]
    cleaned, size_type = extract_size_type(cleaned)
    joined = "".join(cleaned)

    exact = replace(cfg, max_confusions=0, allow_infer=False)
    readings: list[CodeReading] = []
    for cand, speculative in candidate_strings(cleaned, cfg):
        reading = read_candidate(cand, exact if speculative else cfg)
        if speculative and reading.status != VERIFIED:
            continue
        readings.append(reading)

    usable = [r for r in readings if r.code]
    if not usable:
        note = next((r.note for r in readings if r.note), "")
        return CodeReading(None, FAILED, joined, 0, size_type=size_type, note=note)

    top_rank = max(STATUS_RANK[r.status] for r in usable)
    top = [r for r in usable if STATUS_RANK[r.status] == top_rank]
    if cfg.reject_conflicts and len({r.code for r in top}) > 1:
        return CodeReading(None, FAILED, joined, 0, size_type=size_type,
                           note=f"conflict: {len({r.code for r in top})} different "
                                f"{top[0].status} codes")
    best = min(top, key=lambda r: r.edits)    # min() keeps priority order on ties
    return replace(best, size_type=size_type)
