"""Read container codes on every tracked frame of every event (milestone M2).

Third stage of the event extractor, after scripts/track_trucks.py and
scripts/extract_events.py. For each event (rebuilt from tracks.csv with
the same code as extract_events.py, so numbering matches events.csv) it
takes every tracked frame on the cameras whose role is "codes" or
"backup" (558, 559), crops the truck box, and runs PaddleOCR on the GPU
at all four rotations.

Every rotation's reading is stored, not just the best, so that
- the per-camera rotation question (is one fixed rotation enough?) can be
  answered from this file without running OCR again, and
- fusion (src/fusion.py, scripts/fuse_events.py) can be re-run cheaply
  with different rules.

Whole-crop OCR, no CCLN: on the footage the truck crop already isolates
the container, and CCLN (trained on close photos) localised worse than
reading the whole crop (truck scan, Oct 2026).

Frames where the truck's box is small (longest side < --min-box-px) are
skipped: the truck is still far away and the code is unreadable.

Output (results/ is gitignored):
    results/events/reads.csv   one row per (frame, rotation)

Usage:
    python scripts/read_events.py
    python scripts/read_events.py --events 2 4      # only some events
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.events import extract_events
from src.footage import load_camera_configs, load_tracks_csv, parse_recording_name

ROTATIONS = (0, 90, 180, 270)
READ_ROLES = ("codes", "backup")


def crop_box(frame, box, pad_frac=0.10):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    return frame[max(0, int(y1 - py)):min(h, int(y2 + py)),
                 max(0, int(x1 - px)):min(w, int(x2 + px))]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--tracks", default="results/events/tracks.csv")
    parser.add_argument("--out", default="results/events/reads.csv")
    parser.add_argument("--events", nargs="*", type=int, default=None)
    parser.add_argument("--min-box-px", type=float, default=600)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    configs = load_camera_configs(args.config)
    videos = {info.camera: p for p in Path(args.videos_dir).glob("*.mp4")
              if (info := parse_recording_name(p.name))}
    infos = {cam: parse_recording_name(p.name) for cam, p in videos.items()}
    ex = extract_events(load_tracks_csv(args.tracks), configs, infos)
    read_cams = [c for c in ex.chain if configs[c].role in READ_ROLES]

    # (camera -> [(event no, Detection), ...]) so each video is opened once.
    todo = defaultdict(list)
    for i, e in enumerate(ex.events, 1):
        if args.events and i not in args.events:
            continue
        for cam in read_cams:
            for p in e.on(cam):
                for d in ex.detections[(cam, p.track_id)]:
                    if d.longest_side >= args.min_box_px:
                        todo[cam].append((i, d))
    n_frames = sum(len(v) for v in todo.values())
    print(f"{len(ex.events)} events; reading {n_frames} frames on cameras "
          f"{', '.join(read_cams)} at {len(ROTATIONS)} rotations")

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    import cv2
    from PIL import Image
    from src.iso6346_recovery import read_container_code
    from src.ocr import RealOCRAdapter

    ocr = RealOCRAdapter(paddle_gpu=not args.cpu, use_localization=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "camera", "track_id", "frame", "t", "box_px", "rotation",
                         "status", "code", "size_type", "edits", "mean_conf", "raw", "note"])
        for cam, items in todo.items():
            cap = cv2.VideoCapture(str(videos[cam]))
            for event_no, d in sorted(items, key=lambda x: x[1].frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, d.frame)
                ok, frame = cap.read()
                if not ok:
                    continue
                crop = Image.fromarray(cv2.cvtColor(crop_box(frame, d.box), cv2.COLOR_BGR2RGB))
                for rot in ROTATIONS:
                    img = crop if rot == 0 else crop.rotate(rot, expand=True)
                    fragments = ocr._ocr_one(img)
                    r = read_container_code(fragments)
                    confs = [c for _, c in fragments]
                    writer.writerow([
                        event_no, cam, d.track_id, d.frame, round(d.t, 2),
                        round(d.longest_side), rot, r.status, r.code or "", r.size_type or "",
                        r.edits, round(sum(confs) / len(confs), 3) if confs else 0,
                        r.raw, r.note,
                    ])
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{n_frames} frames ({time.time() - t0:.0f} s)", flush=True)
            cap.release()
    print(f"Wrote {out} ({done} frames, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
