"""Fuse per-frame readings into one container code per event (milestone M2).

Reads results/events/reads.csv (scripts/read_events.py: every frame of
every event on 558/559, at all four rotations) and, without running OCR
again:

1. Rotation: for each camera, compares "best of all four rotations" with
   each single fixed rotation — frames read validated, and whether the
   fused code per event changes. A fixed rotation is 4x faster, which
   matters for the dashboard replay; this table says whether it is safe.
2. Fusion: per event, fuses the frames (best rotation per frame) per
   camera (multi-frame) and over all cameras (cross-camera) with
   src.fusion.StatusWeightedVoteFuser.

Codes here are predictions; nothing is scored against ground truth until
the M3 labels exist (M4).

Output (results/ is gitignored):
    results/events/fused.csv     one row per event

Usage:
    python scripts/fuse_events.py
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

from src.fusion import FrameRead, StatusWeightedVoteFuser, best_reading
from src.iso6346_recovery import STATUS_RANK, RECOVERED

ROTATIONS = (0, 90, 180, 270)


def load_reads(path):
    """{(event, camera, frame): [FrameRead per rotation]}"""
    by_frame = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            by_frame[(int(r["event"]), r["camera"], int(r["frame"]))].append(FrameRead(
                camera=r["camera"], frame=int(r["frame"]), status=r["status"],
                code=r["code"] or None, size_type=r["size_type"] or None,
                edits=int(r["edits"]), conf=float(r["mean_conf"]), rotation=int(r["rotation"]),
            ))
    return by_frame


def frames_for(by_frame, rotation=None):
    """{(event, camera): [FrameRead]} using the best rotation, or one fixed rotation."""
    out = defaultdict(list)
    for (event, cam, _), reads in sorted(by_frame.items()):
        pool = reads if rotation is None else [r for r in reads if r.rotation == rotation]
        if pool:
            out[(event, cam)].append(best_reading(pool))
    return out


def validated(r: FrameRead) -> bool:
    return STATUS_RANK[r.status] >= STATUS_RANK[RECOVERED]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reads", default="results/events/reads.csv")
    parser.add_argument("--events", default="results/events/events.csv")
    parser.add_argument("--out", default="results/events/fused.csv")
    args = parser.parse_args()

    fuser = StatusWeightedVoteFuser()
    by_frame = load_reads(args.reads)
    cameras = sorted({cam for _, cam, _ in by_frame})
    with open(args.events) as f:
        events = {int(r["event"]): r for r in csv.DictReader(f)}

    # --- 1. rotation policies, per camera ------------------------------------
    best = frames_for(by_frame)
    fused_best = {k: fuser.fuse(v) for k, v in best.items()}
    print("Rotation per camera (frames = frames read; validated = verified/recovered):")
    print(f"  {'camera':6} {'rotation':>9} {'validated':>10} {'events w/ code':>15} "
          f"{'same code as best-of-4':>23}")
    for cam in cameras:
        keys = [k for k in best if k[1] == cam]
        n_frames = sum(len(best[k]) for k in keys)
        for rot in (None,) + ROTATIONS:
            fr = best if rot is None else frames_for(by_frame, rot)
            n_val = sum(validated(r) for k in keys for r in fr[k])
            fused = {k: fuser.fuse(fr[k]) for k in keys}
            with_code = sum(1 for k in keys if fused[k].code)
            same = sum(1 for k in keys if fused[k].code == fused_best[k].code)
            label = "best-of-4" if rot is None else f"{rot}°"
            print(f"  {cam:6} {label:>9} {n_val:>4}/{n_frames:<5} {with_code:>8}/{len(keys):<6} "
                  f"{same:>15}/{len(keys)}")
        wins = defaultdict(int)
        for k in keys:
            for r in best[k]:
                if validated(r):
                    wins[r.rotation] += 1
        print(f"  {cam:6} validated reads by winning rotation: "
              + ", ".join(f"{rot}°: {wins[rot]}" for rot in ROTATIONS))

    # --- 2. fusion per event ---------------------------------------------------
    scopes = cameras + ["all"]
    rows = []
    for ev in sorted(events):
        row = {"event": ev, "entry_time": events[ev]["entry_time"], "cameras": events[ev]["cameras"]}
        all_frames = []
        for cam in cameras:
            frames = best.get((ev, cam), [])
            all_frames += frames
            row[f"{cam}_frames"] = len(frames)
            row[f"{cam}_frames_validated"] = sum(validated(r) for r in frames)
        for scope in scopes:
            frames = all_frames if scope == "all" else best.get((ev, scope), [])
            out = fuser.fuse(frames)
            row[f"{scope}_code"] = out.code or ""
            row[f"{scope}_status"] = out.status if frames else ""
            row[f"{scope}_support"] = out.support
            row[f"{scope}_size_type"] = out.size_type or ""
            row[f"{scope}_note"] = out.note
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nFused code per event (cross-camera; per-camera codes in fused.csv):")
    for r in rows:
        frames = "/".join(f"{r[f'{c}_frames_validated']}v{r[f'{c}_frames']}" for c in cameras)
        per_cam = " ".join(f"{c}:{r[f'{c}_code'] or '-'}" for c in cameras)
        print(f"  {r['event']:3d} {r['entry_time'][11:]} {r['all_code'] or '-':12} "
              f"{r['all_status']:9} {r['all_size_type'] or '-':5} sup={r['all_support']:<3} "
              f"frames {frames:9} | {per_cam} {r['all_note']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
