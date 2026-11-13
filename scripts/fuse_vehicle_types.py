"""Fuse per-frame vehicle types per truck and score them against the labels (FR3, M7).

Input: results/events/vehicle_type_frames.csv (scripts/classify_events.py)
and the label sheet's vehicle_type column (results/events/labels/labels.csv).

Methods compared, as for FR1 and FR4:
    single frame 561   the one cab-camera frame with the largest box
    fused <camera>     mean probabilities over that camera's frames only
    fused all cameras  mean over every frame on every camera (the M7 output)
A truck is "unknown" when the winning mean probability is below
--min-conf (0.6, from the hand-off notes); unknown is reported separately
from wrong — it goes to a manual check instead of a forced class.

Also reported, because the hand-off flagged it: per-frame accuracy split
by crop shape. The classifier centre-crops to a square, so very wide
side-on crops (width > 2x height) lose the cab and the trailer ends.

The footage is a held-out test: the classifier never saw it. But it is
small (16 vehicles, 3 of the 5 classes, no tanker), so the per-truck
numbers are an indication, not a measurement.

Outputs (results/ is gitignored):
    results/events/vehicle_types.csv   one row per event
    results/events/fr3_results.md      tables for the report
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.vehicle_type import (
    CLASSES, UNKNOWN, fuse_vehicle_type, label_to_class, load_frame_types, single_frame_type,
)

CAMERA_ORDER = ("561", "557", "560", "558", "559")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", default="results/events/vehicle_type_frames.csv")
    parser.add_argument("--labels", default="results/events/labels/labels.csv")
    parser.add_argument("--out", default="results/events/vehicle_types.csv")
    parser.add_argument("--report", default="results/events/fr3_results.md")
    parser.add_argument("--min-conf", type=float, default=0.6)
    args = parser.parse_args()

    frames = load_frame_types(args.frames)
    truth = {}
    if Path(args.labels).exists():                 # new recordings have no labels yet
        with open(args.labels) as f:
            truth = {int(r["event"]): label_to_class(r["vehicle_type"]) for r in csv.DictReader(f)}
    events = sorted(set(frames) | set(truth))
    labelled = [e for e in events if truth.get(e)]

    methods = {"single frame 561": lambda fr: single_frame_type(fr, "561", min_conf=args.min_conf)}
    for cam in CAMERA_ORDER:
        methods[f"fused {cam}"] = lambda fr, cam=cam: fuse_vehicle_type(fr, cameras=[cam],
                                                                           min_conf=args.min_conf)
    methods["fused all cameras"] = lambda fr: fuse_vehicle_type(fr, min_conf=args.min_conf)
    methods["fused all cameras, cameras weighted equally*"] = lambda fr: fuse_vehicle_type(
        fr, min_conf=args.min_conf, per_camera=True)

    results = {name: {e: fn(frames.get(e, [])) for e in events} for name, fn in methods.items()}
    final = results["fused all cameras"]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "vehicle_type", "conf", "top", "frames", "cameras", "label",
                    *[f"mean_{c}" for c in CLASSES]])
        for e in events:
            r = final[e]
            w.writerow([e, r.vehicle_type, round(r.conf, 3), r.top or "", r.frames, "+".join(r.cameras),
                        truth.get(e) or "", *(round(r.mean.get(c, 0), 3) for c in CLASSES)])

    lines = ["# FR3 vehicle type on the depot footage (M7)", "",
             f"{len(labelled)} labelled vehicles: "
             + ", ".join(f"{n} {c}" for c, n in Counter(truth[e] for e in labelled).most_common())
             + f". Unknown = mean probability of the top class below {args.min_conf}.",
             "The classifier never saw this footage (held-out), but 16 vehicles and 3 of 5 "
             "classes are too few for a real accuracy figure.", "",
             "## Per truck", "",
             "| Method | Trucks with frames | Correct | Wrong | Unknown |",
             "|---|---|---|---|---|"]
    for name, res in results.items():
        have = [e for e in labelled if res[e].frames]
        c = sum(res[e].vehicle_type == truth[e] for e in have)
        u = sum(res[e].vehicle_type == UNKNOWN for e in have)
        lines.append(f"| {name} | {len(have)} | {c}/{len(have)} | {len(have) - c - u} | {u} |")
    lines += ["", "\\* Chosen after seeing the frame-weighted results on this footage (a truck "
              "waiting under 560 outvoted the other cameras), so it is tuned on the test set: "
              "optimistic, to be confirmed on new footage."]

    lines += ["", "## Per truck, all cameras fused", "",
              "| Event | Label | Predicted | Mean prob | Top class | Frames | Cameras |",
              "|---|---|---|---|---|---|---|"]
    for e in events:
        r = final[e]
        mark = "" if r.vehicle_type == truth.get(e) else " ✗"
        lines.append(f"| {e} | {truth.get(e) or '—'} | {r.vehicle_type}{mark} | {r.conf:.2f} | "
                     f"{r.top or '—'} | {r.frames} | {'+'.join(r.cameras)} |")

    present = [c for c in CLASSES if any(truth[e] == c for e in labelled)]
    cols = list(CLASSES) + [UNKNOWN]
    conf_m = defaultdict(Counter)
    for e in labelled:
        conf_m[truth[e]][final[e].vehicle_type] += 1
    lines += ["", "## Confusion (rows = label, columns = predicted), all cameras fused", "",
              "| label \\ predicted | " + " | ".join(cols) + " |",
              "|---|" + "---|" * len(cols)]
    for c in present:
        lines.append(f"| {c} | " + " | ".join(str(conf_m[c][p]) for p in cols) + " |")

    # Per-frame accuracy by camera and by crop shape (uses the raw CSV for crop sizes).
    shape = {}
    with open(args.frames) as f:
        for r in csv.DictReader(f):
            shape[(int(r["event"]), r["camera"], int(r["frame"]))] = int(r["crop_w"]) / max(1, int(r["crop_h"]))
    by_cam, by_shape = defaultdict(Counter), defaultdict(Counter)
    for e in labelled:
        for fr in frames.get(e, []):
            ok = fr.top == truth[e]
            wide = "wide (w > 2h)" if shape.get((e, fr.camera, fr.frame), 1) > 2 else "not wide"
            for d, k in ((by_cam, fr.camera), (by_shape, wide)):
                d[k]["n"] += 1
                d[k]["ok"] += ok
    lines += ["", "## Per frame (top class of each frame, labelled vehicles)", "",
              "| Camera | Frames | Top class correct |", "|---|---|---|"]
    for cam in CAMERA_ORDER:
        if by_cam[cam]["n"]:
            n, ok = by_cam[cam]["n"], by_cam[cam]["ok"]
            lines.append(f"| {cam} | {n} | {ok / n:.0%} |")
    lines += ["", "| Crop shape | Frames | Top class correct |", "|---|---|---|"]
    for k in ("not wide", "wide (w > 2h)"):
        if by_shape[k]["n"]:
            n, ok = by_shape[k]["n"], by_shape[k]["ok"]
            lines.append(f"| {k} | {n} | {ok / n:.0%} |")

    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
