"""Fuse licence plates per event and score them against labels (milestone M6).

Reads results/events/plate_fragments.csv (scripts/read_plates.py), turns
each frame's OCR fragments into plate candidates (src.plates) and fuses,
per event, one vehicle plate (prime mover) and one trailer plate ("T/…").

Scoring uses the `plate` column of labels.csv for the vehicle plate and
`trailer_plate` when that column exists. A blank label means the
labeller could not read the plate — the truth is unknown, so that event
is left out of the score (not counted as "no plate"); its prediction is
listed for checking instead.

Rows: one frame per truck (561, largest truck box — what the label sheet
first showed), multi-frame per camera, and all cameras fused.

Outputs (results/ is gitignored — plates are personal data):
    results/events/plates.csv          per event: fused plates + best evidence frame
    results/events/fr4_results.md      the table + per-event detail

Usage:
    python scripts/fuse_plates.py
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ocr_eval import Tally
from src.plates import TRAILER, VEHICLE, fuse_plates, load_plate_fragments, plates_in_frame


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragments", default="results/events/plate_fragments.csv")
    parser.add_argument("--labels", default="results/events/labels/labels.csv")
    parser.add_argument("--events", default="results/events/events.csv")
    parser.add_argument("--out-dir", default="results/events")
    args = parser.parse_args()

    frags = load_plate_fragments(args.fragments)
    with open(args.events) as f:
        events = [int(r["event"]) for r in csv.DictReader(f)]
    with open(args.labels) as f:
        labels = {int(r["event"]): r for r in csv.DictReader(f)}
    truth = {VEHICLE: {e: r.get("plate", "").replace(" ", "").upper() for e, r in labels.items()},
             TRAILER: {e: r.get("trailer_plate", "").replace(" ", "").upper() for e, r in labels.items()}}

    reads = defaultdict(list)                     # (event, camera) -> [PlateRead]
    frame_sizes = {}
    for (ev, cam, fr), fs in frags.items():
        reads[(ev, cam)] += plates_in_frame(fs, camera=cam, frame=fr)
    with open(args.fragments) as f:
        for r in csv.DictReader(f):
            frame_sizes[(int(r["event"]), r["camera"], int(r["frame"]))] = float(r["box_px"])
    cameras = sorted({cam for _, cam in reads} | {cam for _, cam, _ in frags})

    rows, table = [], []
    for ev in events:
        all_reads = [r for cam in cameras for r in reads.get((ev, cam), [])]
        row = {"event": ev}
        for kind in (VEHICLE, TRAILER):
            out = fuse_plates(all_reads, kind)
            e = out.evidence
            row.update({f"{kind}_plate": out.plate or "", f"{kind}_support": out.support,
                        f"{kind}_cameras": "+".join(out.cameras), f"{kind}_note": out.note,
                        f"{kind}_evidence": (f"{e.camera}:{e.frame}:" + ",".join(f"{v:.4f}" for v in e.box))
                        if e else ""})
        rows.append(row)

    def score(kind, method, scope, pred_for):
        tally = Tally()
        for ev in events:
            t = truth[kind].get(ev, "")
            if not t:
                continue                          # unreadable to the labeller: unknown truth
            pred = pred_for(ev)
            if pred is not ...:
                tally.add(pred, t)
        table.append((kind, method, scope, tally))

    for kind in (VEHICLE, TRAILER):
        if not any(truth[kind].values()):
            continue
        # one frame: 561, largest truck box (the frame the M3 sheet showed)
        def one_frame(ev, kind=kind):
            fr = [k for k in frags if k[0] == ev and k[1] == "561"]
            if not fr:
                return ...
            best = max(fr, key=lambda k: frame_sizes[k])
            hits = [r.plate for r in plates_in_frame(frags[best]) if r.kind == kind]
            return hits[0] if hits else None
        score(kind, "one frame (largest box)", "561", one_frame)
        for cam in cameras:
            score(kind, "multi-frame", cam,
                  lambda ev, cam=cam, kind=kind: fuse_plates(reads.get((ev, cam), []), kind).plate
                  if (ev, cam) in reads or any(k[:2] == (ev, cam) for k in frags) else ...)
        score(kind, "cross-camera", "all cameras",
              lambda ev, kind=kind: fuse_plates([r for c in cameras for r in reads.get((ev, c), [])],
                                                kind).plate)

    out_dir = Path(args.out_dir)
    with (out_dir / "plates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# FR4 — licence plates on the depot footage (baseline: OCR + format check + vote)", "",
             "Scored only where the labeller could read the plate; blank labels are unknown truth. "
             "Correct = the plate output equals the label; wrong = a different plate; "
             "missed = no plate output.", "",
             "| plate | method | scope | labelled | correct | wrong | missed | char acc |",
             "|---|---|---|---|---|---|---|---|"]
    for kind, method, scope, t in table:
        c = t.counts
        lines.append(f"| {kind} | {method} | {scope} | {t.with_container} | "
                     f"{c['correct']} ({pct(t.rate('correct'))}) | {c['wrong']} ({pct(t.rate('wrong'))}) | "
                     f"{c['missed']} ({pct(t.rate('missed'))}) | {pct(t.mean_char_accuracy)} |")
    lines += ["", "## Per event (all cameras fused)", "",
              "| event | vehicle label | vehicle pred (frames) | trailer label | trailer pred (frames) |",
              "|---|---|---|---|---|"]
    for r in rows:
        ev = r["event"]
        lines.append(f"| {ev} | {truth[VEHICLE].get(ev) or '?'} | "
                     f"{r['vehicle_plate'] or '–'} ({r['vehicle_support']}) | "
                     f"{truth[TRAILER].get(ev) or '?'} | {r['trailer_plate'] or '–'} ({r['trailer_support']}) |")
    (out_dir / "fr4_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'plates.csv'} and {out_dir / 'fr4_results.md'}")


if __name__ == "__main__":
    main()
