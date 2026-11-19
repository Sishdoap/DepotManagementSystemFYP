"""Classify the vehicle type on every tracked frame of every event (milestone M7).

Same structure as scripts/read_events.py: events are rebuilt from
tracks.csv with the same code as extract_events.py (so numbering matches
events.csv and the label sheet), every tracked frame of each event is
cropped (detector box + 10% per side, src.footage.crop_box — the training
crop) and the classifier in src/vehicle_type.py gives class probabilities.
Per-truck fusion and scoring are cheap and separate
(scripts/fuse_vehicle_types.py), so every frame's probabilities are kept.

All cameras except role "ignore" are classified, including the side
cameras: whether the close, very wide side-on crops classify as well as
the cab view is one of the things M7 measures (the classifier centre-crops
to a square, so a wide crop loses its ends).

Frames whose box is smaller than --min-side-px on its shorter side are
skipped (far away; the hand-off notes use 64 px).

Only the tracker's detections are classified: the tracker follows COCO
truck/bus/train (scripts/track_trucks.py). Cars that YOLO labels "car"
are never tracked, so light vehicles reach FR3 only when YOLO calls them a
truck (both light vehicles on the footage do). Adding "car" to the tracker
would renumber every event and invalidate the labels, so it is left for
new footage.

Output (results/ is gitignored):
    results/events/vehicle_type_frames.csv   one row per (event, camera, frame)

Usage:
    python scripts/classify_events.py
    python scripts/classify_events.py --events 3 5 --out /tmp/vt.csv
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.events import extract_events
from src.footage import crop_box, find_recordings, load_camera_configs, load_tracks_csv, parse_recording_name
from src.vehicle_type import CLASSES, UltralyticsVehicleTypeClassifier

COLUMNS = ["event", "camera", "track_id", "frame", "t", "box_px", "crop_w", "crop_h", *CLASSES]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--tracks", default="results/events/tracks.csv")
    parser.add_argument("--weights", default="models/vehicle_type.pt")
    parser.add_argument("--card", default="models/model_card.json")
    parser.add_argument("--out", default="results/events/vehicle_type_frames.csv")
    parser.add_argument("--events", nargs="*", type=int, default=None)
    parser.add_argument("--min-side-px", type=float, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="e.g. cpu or 0 (default: Ultralytics picks)")
    args = parser.parse_args()

    configs = load_camera_configs(args.config, videos_dir=args.videos_dir)
    videos = find_recordings(args.videos_dir)
    infos = {cam: parse_recording_name(p.name) for cam, p in videos.items()}
    ex = extract_events(load_tracks_csv(args.tracks), configs, infos)
    cams = [c for c in ex.chain if configs[c].role != "ignore"]

    todo = defaultdict(list)
    for i, e in enumerate(ex.events, 1):
        if args.events and i not in args.events:
            continue
        for cam in cams:
            for p in e.on(cam):
                for d in ex.detections[(cam, p.track_id)]:
                    x1, y1, x2, y2 = d.box
                    if min(x2 - x1, y2 - y1) >= args.min_side_px:
                        todo[cam].append((i, d))
    n_frames = sum(len(v) for v in todo.values())
    print(f"{len(ex.events)} events; classifying {n_frames} frames on cameras {', '.join(cams)}")

    import cv2

    clf = UltralyticsVehicleTypeClassifier(args.weights, args.card, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        def flush(batch):
            for (event_no, cam, d, crop), p in zip(batch, clf.predict_batch([b[3] for b in batch])):
                writer.writerow([event_no, cam, d.track_id, d.frame, round(d.t, 2),
                                 round(d.longest_side), crop.shape[1], crop.shape[0],
                                 *(round(p[c], 4) for c in CLASSES)])

        for cam, items in todo.items():
            cap = cv2.VideoCapture(str(videos[cam]))
            batch = []
            for event_no, d in sorted(items, key=lambda x: x[1].frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, d.frame)
                ok, frame = cap.read()
                if not ok:
                    continue
                crop = cv2.cvtColor(crop_box(frame, d.box)[0], cv2.COLOR_BGR2RGB)
                batch.append((event_no, cam, d, crop))
                if len(batch) == args.batch:
                    flush(batch)
                    batch = []
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{n_frames} frames ({time.time() - t0:.0f} s)", flush=True)
            if batch:
                flush(batch)
            cap.release()
    print(f"Wrote {out} ({done} frames, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
