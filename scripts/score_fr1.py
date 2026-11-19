"""FR1 results table: container-code reading on the footage, scored against labels (M5).

Inputs:
    results/events/reads.csv          every frame of every event, all rotations
                                      (scripts/read_events.py)
    results/events/labels/labels.csv  user-checked labels (M3); container_code
                                      blank = no container

Rows (each at the raw / recovered / inferred trust levels, src.ocr_eval):

    per frame       every frame read on a camera, scored on its own
                    (units are frames, not trucks)
    one frame       one frame per truck per camera — the one with the
                    largest truck box, i.e. what a pipeline without fusion
                    would pick
    multi-frame     all frames of one camera fused (src.fusion)
    cross-camera    all frames of several cameras fused: the side pair
                    (557+560), the end-on pair (558+559), and all cameras

Per-camera rows count only the trucks that camera saw ("seen" column);
cross-camera "all" covers every event. Containers and no-container
vehicles are counted separately: for a vehicle without a container the
only right answer is no code. Labels may hold two codes (2x20 ft); fused
rows use src.fusion.MultiContainerFuser followed by
src.fusion.resolve_shared_codes (a container goes to one of the trucks
close in time), and a second table compares the multi-container fuser
alone with the single-code StatusWeightedVoteFuser on all cameras.

Outputs (results/ is gitignored):
    results/events/fr1_results.md          the table, plus per-event detail
    results/events/fr1_predictions.csv     per event x scope x level

Usage:
    python scripts/score_fr1.py
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.fusion import (MultiContainerFuser, StatusWeightedVoteFuser, load_frame_reads,
                        resolve_shared_codes)
from src.iso6346_recovery import FAILED
from src.ocr_eval import LEVELS, Tally, best_at_level

SCOPES = {                       # cross-camera scopes; per-camera scopes are added automatically
    "side 557+560": ("557", "560"),
    "end-on 558+559": ("558", "559"),
}


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reads", default="results/events/reads.csv")
    parser.add_argument("--labels", default="results/events/labels/labels.csv")
    parser.add_argument("--out-dir", default="results/events")
    parser.add_argument("--events", default="results/events/events.csv",
                        help="Event entry times, for src.fusion.resolve_shared_codes.")
    parser.add_argument("--min-support", type=int, default=None,
                        help="Frames a code needs whatever its status (default: the fuser's).")
    args = parser.parse_args()
    with open(args.events) as f:
        times = {int(r["event"]): datetime.fromisoformat(r["entry_time"]) for r in csv.DictReader(f)}

    with open(args.labels) as f:
        labels = {int(r["event"]): r for r in csv.DictReader(f) if r["checked"] == "yes"}
    truth = {e: frozenset(r["container_code"].upper().split()) for e, r in labels.items()}
    truth_st = {e: (r["size_type"].strip().upper() or None) for e, r in labels.items()}

    by_frame = load_frame_reads(args.reads)
    cameras = sorted({cam for _, cam, _ in by_frame})
    # {(event, camera): [[rotation reads of frame 1], [frame 2], ...]}
    frames = defaultdict(list)
    for (ev, cam, _), reads in sorted(by_frame.items()):
        if ev in truth:
            frames[(ev, cam)].append(reads)
    scopes = {cam: (cam,) for cam in cameras}
    scopes.update({k: v for k, v in SCOPES.items() if all(c in cameras for c in v)})
    scopes["all cameras"] = tuple(cameras)

    fuser = (MultiContainerFuser() if args.min_support is None else
             MultiContainerFuser(StatusWeightedVoteFuser(min_support=args.min_support)))
    single = (StatusWeightedVoteFuser() if args.min_support is None else
              StatusWeightedVoteFuser(min_support=args.min_support))
    table, predictions, comparison = [], [], []
    for level in LEVELS:
        # per frame + one frame, per camera
        for cam in cameras:
            per_frame, one_frame = Tally(), Tally()
            for ev in sorted(truth):
                fr = frames.get((ev, cam), [])
                if not fr:
                    continue
                for reads in fr:
                    best = best_at_level(reads, level)
                    per_frame.add(best.code if best.status != FAILED else None, truth[ev])
                largest = max(fr, key=lambda reads: reads[0].box_px)
                best = best_at_level(largest, level)
                one_frame.add(best.code if best.status != FAILED else None, truth[ev])
            table.append((level, "per frame (frames)", cam, per_frame, None))
            table.append((level, "one frame (largest box)", cam, one_frame, None))
        # fused, per scope
        for name, cams in scopes.items():
            tally, st_ok, st_n = Tally(), 0, 0
            per_event = {}
            for ev in sorted(truth):
                fr = [best_at_level(reads, level) for cam in cams for reads in frames.get((ev, cam), [])]
                if not fr and name != "all cameras":
                    continue                       # this camera never saw the truck
                per_event[ev] = fuser.fuse(fr)
            resolve_shared_codes(per_event, times)
            for ev, fused in per_event.items():
                o = tally.add_vehicle(fused.codes, truth[ev])
                if truth_st[ev]:
                    st_n += 1
                    st_ok += fused.size_type == truth_st[ev]
                predictions.append({"event": ev, "level": level, "scope": name,
                                    "truth": " ".join(sorted(truth[ev])),
                                    "pred": " ".join(fused.codes),
                                    "status": fused.status, "support": fused.support,
                                    "frames": fused.frames, "outcome": o,
                                    "size_type_truth": truth_st[ev] or "",
                                    "size_type_pred": fused.size_type or "", "note": fused.note})
            method = "multi-frame" if len(cams) == 1 else "cross-camera"
            table.append((level, method, name, tally, (st_ok, st_n)))
        # single-code vs multi-container fuser, all cameras
        for label, f in (("single code (StatusWeightedVote)", single),
                         ("multi-container", fuser)):
            tally = Tally()
            for ev in sorted(truth):
                fr = [best_at_level(reads, level) for cam in scopes["all cameras"]
                      for reads in frames.get((ev, cam), [])]
                tally.add_vehicle(f.fuse(fr).codes, truth[ev])
            comparison.append((level, label, tally))

    n_two = sum(1 for t in truth.values() if len(t) > 1)
    lines = ["# FR1 — container code reading on the depot footage", "",
             f"{sum(1 for t in truth.values() if t)} trucks with a container "
             f"({n_two} carrying two 20 ft containers), "
             f"{sum(1 for t in truth.values() if not t)} vehicles without one "
             f"(labels: {args.labels}). Per-frame rows count frames (a frame is correct "
             "if it reads any of the truck's codes); all other rows count vehicles "
             "(correct = all codes, nothing extra; partial = some codes, none wrong).", "",
             "| level | method | scope | seen (cont / none) | correct | partial | wrong | missed | "
             "false code | char acc | size/type |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cells(t):
        c = t.counts
        return (f"{c['correct']} ({pct(t.rate('correct'))}) | {c['partial']} | "
                f"{c['wrong']} ({pct(t.rate('wrong'))}) | {c['missed']} ({pct(t.rate('missed'))}) | "
                f"{c['false_code']} | {pct(t.mean_char_accuracy)}")

    for level, method, scope, t, st in table:
        st_txt = f"{st[0]}/{st[1]}" if st else "–"
        lines.append(f"| {level} | {method} | {scope} | {t.with_container} / {t.without_container} | "
                     f"{cells(t)} | {st_txt} |")
    lines += ["", "## Single-code vs multi-container fusion (all cameras)", "",
              "| level | fuser | correct | partial | wrong | missed | false code | char acc |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {level} | {label} | {cells(t)} |" for level, label, t in comparison]
    out_dir = Path(args.out_dir)
    (out_dir / "fr1_results.md").write_text("\n".join(lines) + "\n")
    with (out_dir / "fr1_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'fr1_results.md'} and {out_dir / 'fr1_predictions.csv'}")


if __name__ == "__main__":
    main()
