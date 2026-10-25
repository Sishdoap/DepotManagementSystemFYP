"""Offline ablation of container-code reading on captured OCR fragments.

Re-runs the code-reading step — no OCR, no models — on the per-rotation
fragments saved by:

    python scripts/benchmark_ocr.py --all-rotations --csv-output results/ocr_capture_all_rotations.csv

and compares the FYP1 recovery (frozen copy below) against several
configurations of src/iso6346_recovery.py. Rotation selection mirrors
RealOCRAdapter: rotations are tried in order 0/90/180/270 and the search
stops at the first validated read.

The key column is "validated but wrong": codes the pipeline claims are
confirmed by the check digit but don't match the label. Those are silent
errors, so a configuration is only better if it adds correct reads without
adding those.

Usage:
    python scripts/replay_recovery.py
    python scripts/replay_recovery.py --capture results/ocr_capture_all_rotations.csv
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.iso6346 import validate
from src.iso6346_recovery import (
    DIGIT_CONFUSIONS,
    FAILED,
    LETTER_CONFUSIONS,
    INFERRED,
    RECOVERED,
    VERIFIED,
    ReaderConfig,
    read_container_code,
    reading_key,
    symmetric_pairs,
)

ROTATIONS = ("0", "90", "180", "270")


# --- FYP1 baseline (frozen copy of the pre-FYP2 logic, for comparison) ---

_LEGACY_RE = re.compile(r"[A-Z]{4}\d{7}")
_L_FROM_D = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "7": "T", "8": "B"}
_D_FROM_L = {v: k for k, v in _L_FROM_D.items()}
_D_FROM_L.update({"D": "0", "Q": "0", "A": "4"})


def _legacy_valid(code):
    return bool(_LEGACY_RE.fullmatch(code)) and validate(code).is_valid


def _legacy_correct(text, max_subs=3):
    if len(text) != 11:
        return None
    slots = []
    for i, ch in enumerate(text):
        if i < 4:
            out = {ch} | ({_L_FROM_D[ch]} if ch in _L_FROM_D else set())
            slots.append(list(out) if ch.isalpha() else [_L_FROM_D.get(ch, ch)])
        else:
            out = {ch} | ({_D_FROM_L[ch]} if ch in _D_FROM_L else set())
            slots.append(list(out) if ch.isdigit() else [_D_FROM_L.get(ch, ch)])
    for combo in product(*slots):
        if sum(a != b for a, b in zip(combo, text)) > max_subs:
            continue
        cand = "".join(combo)
        if _legacy_valid(cand):
            return cand
    return None


def legacy_read(fragments):
    """(code, validated, score) — port of RealOCRAdapter._best_from_fragments."""
    cleaned = [(re.sub(r"[^A-Z0-9]", "", t.upper()), c) for t, c in fragments]
    cleaned = [(t, c) for t, c in cleaned if t]
    if not cleaned:
        return None, False, 0.0
    joined = "".join(t for t, _ in cleaned)
    mean_conf = sum(c for _, c in cleaned) / len(cleaned)
    cands = [m.group() for t, _ in cleaned for m in _LEGACY_RE.finditer(t)]
    cands += [m.group() for m in _LEGACY_RE.finditer(joined)]
    for code in cands:
        if _legacy_valid(code):
            return code, True, 1000 + mean_conf
    for code in cands:
        fixed = _legacy_correct(code)
        if fixed:
            return fixed, True, 900 + mean_conf
    for i in range(max(0, len(joined) - 10)):
        fixed = _legacy_correct(joined[i:i + 11])
        if fixed:
            return fixed, True, 800 + mean_conf
    return None, False, mean_conf


def legacy_pipeline(by_rot):
    best = (None, False, -1.0)
    for rot in ROTATIONS:
        if rot not in by_rot:
            continue
        code, valid, score = legacy_read(by_rot[rot])
        if score > best[2]:
            best = (code, valid, score)
        if valid:
            break
    code, valid, _ = best
    return code, (VERIFIED if valid else FAILED)


# --- New pipeline ---

def new_pipeline(by_rot, cfg):
    """Mirror of RealOCRAdapter._recognize on saved fragments."""
    best, best_key = None, None
    for rot in ROTATIONS:
        if rot not in by_rot:
            continue
        frags = by_rot[rot]
        reading = read_container_code(frags, cfg)
        key = reading_key(reading, frags)
        if best_key is None or key > best_key:
            best, best_key = reading, key
        if reading.is_validated:
            break
    return best.code, best.status


_FORCED = ReaderConfig(max_confusions=0, try_check_digit_first=False)

CONFIGS = {
    "forced fixes only": _FORCED,
    "  ... without conflict rejection": replace(_FORCED, reject_conflicts=False),
    "  ... + check-digit-first": replace(_FORCED, try_check_digit_first=True),
    "  ... + 0/8 confusion": replace(_FORCED, max_confusions=1,
                                     digit_confusions=symmetric_pairs([("0", "8")]),
                                     letter_confusions={}),
    "  ... + letter confusions": replace(_FORCED, max_confusions=1, digit_confusions={}),
    "  ... + all confusions": replace(_FORCED, max_confusions=1,
                                      digit_confusions=DIGIT_CONFUSIONS,
                                      letter_confusions=LETTER_CONFUSIONS),
    "  ... no inference": replace(_FORCED, allow_infer=False),
}


def tally(results, truths):
    t = {"validated_correct": 0, "validated_wrong": 0, "inferred_correct": 0,
         "inferred_wrong": 0, "failed": 0}
    for (code, status), truth in zip(results, truths):
        if status in (VERIFIED, RECOVERED):
            t["validated_correct" if code == truth else "validated_wrong"] += 1
        elif status == INFERRED:
            t["inferred_correct" if code == truth else "inferred_wrong"] += 1
        else:
            t["failed"] += 1
    return t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", default="results/ocr_capture_all_rotations.csv")
    parser.add_argument("--out", default="results/recovery_ablation.md")
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(open(args.capture)) if r.get("fragments")]
    rows = [r for r in rows if validate(r["ground_truth"]).is_valid]   # skip unverifiable labels
    by_rots = [{k: [tuple(f) for f in v] for k, v in json.loads(r["fragments"]).items()} for r in rows]
    truths = [r["ground_truth"] for r in rows]
    n = len(rows)

    table = {"FYP1 recovery (baseline)": tally([legacy_pipeline(b) for b in by_rots], truths)}
    for name, cfg in CONFIGS.items():
        table[name] = tally([new_pipeline(b, cfg) for b in by_rots], truths)

    lines = [
        "# Container-code reading ablation",
        "",
        f"{n} labelled images, replayed from `{args.capture}` (same OCR output for every row).",
        "",
        "| Configuration | Validated correct | **Validated but wrong** | Inferred correct | Inferred wrong | Failed |",
        "|---|---|---|---|---|---|",
    ]
    for name, t in table.items():
        lines.append(
            f"| {name} | {t['validated_correct']} ({t['validated_correct'] / n:.1%}) | "
            f"**{t['validated_wrong']}** | {t['inferred_correct']} | {t['inferred_wrong']} | {t['failed']} |"
        )
    lines += [
        "",
        "*Validated* = verified or recovered (the check digit confirms the code).",
        "*Inferred* = check digit computed from the other ten characters; unverified.",
    ]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
