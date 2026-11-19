"""OCR every frame of every event for licence plates (milestone M6, baseline).

Baseline without a trained plate detector: for each event (rebuilt from
tracks.csv as in extract_events.py, so numbering matches) it takes every
tracked frame on the cameras that can see a plate — 561 (cab front),
557/560 (cab front on arrival, trailer rear on departure) and 558
(trailer rear) — crops the truck box and runs PaddleOCR once (rotation
0; plates are horizontal). Every text fragment is stored with its box,
as a fraction of the truck box, so plate parsing (src/plates.py) can be
tuned without running OCR again.

Frames where the truck box is small (longest side < --min-box-px) are
skipped: the plate is a few pixels high there.

Output (results/ is gitignored — plates are personal data):
    results/events/plate_fragments.csv   one row per OCR fragment

Usage:
    python scripts/read_plates.py
    python scripts/read_plates.py --events 8 --out /tmp/plates_test.csv
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
from src.footage import crop_box, find_recordings, load_camera_configs, load_tracks_csv, parse_recording_name

PLATE_ROLES = ("cab", "side", "codes")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--tracks", default="results/events/tracks.csv")
    parser.add_argument("--out", default="results/events/plate_fragments.csv")
    parser.add_argument("--events", nargs="*", type=int, default=None)
    parser.add_argument("--min-box-px", type=float, default=600)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--det-max-side", type=int, default=1920,
                        help="Cap on the text-detection image size; 0 = full resolution.")
    parser.add_argument("--gpu-mem-mb", type=int, default=6144,
                        help="Cap on PaddleOCR's GPU memory; 0 = no cap.")
    args = parser.parse_args()

    configs = load_camera_configs(args.config, videos_dir=args.videos_dir)
    videos = find_recordings(args.videos_dir)
    infos = {cam: parse_recording_name(p.name) for cam, p in videos.items()}
    ex = extract_events(load_tracks_csv(args.tracks), configs, infos)
    cams = [c for c in ex.chain if configs[c].role in PLATE_ROLES]

    todo = defaultdict(list)
    for i, e in enumerate(ex.events, 1):
        if args.events and i not in args.events:
            continue
        for cam in cams:
            for p in e.on(cam):
                todo[cam] += [(i, d) for d in ex.detections[(cam, p.track_id)]
                              if d.longest_side >= args.min_box_px]
    n_frames = sum(len(v) for v in todo.values())
    print(f"{len(ex.events)} events; reading {n_frames} frames on cameras {', '.join(cams)}")

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    import cv2
    from src.ocr import RealOCRAdapter

    paddle = RealOCRAdapter._init_paddle(use_gpu=not args.cpu,
                                         det_max_side=args.det_max_side or None,
                                         gpu_mem_mb=args.gpu_mem_mb or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "camera", "track_id", "frame", "t", "box_px",
                         "text", "conf", "x1", "y1", "x2", "y2"])
        for cam, items in todo.items():
            cap = cv2.VideoCapture(str(videos[cam]))
            for event_no, d in sorted(items, key=lambda x: x[1].frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, d.frame)
                ok, frame = cap.read()
                if not ok:
                    continue
                crop, (x0, y0) = crop_box(frame, d.box)
                res = paddle.predict(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))[0]
                bx1, by1, bx2, by2 = d.box
                bw, bh = bx2 - bx1, by2 - by1
                for text, conf, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                    if not text.strip():
                        continue
                    xs = [p[0] + x0 for p in poly]
                    ys = [p[1] + y0 for p in poly]
                    # Fragment box as a fraction of the (unpadded) truck box.
                    writer.writerow([event_no, cam, d.track_id, d.frame, round(d.t, 2),
                                     round(d.longest_side), text, round(float(conf), 3),
                                     *(round(v, 4) for v in ((min(xs) - bx1) / bw, (min(ys) - by1) / bh,
                                                             (max(xs) - bx1) / bw, (max(ys) - by1) / bh))])
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{n_frames} frames ({time.time() - t0:.0f} s)", flush=True)
            cap.release()
    print(f"Wrote {out} ({done} frames, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
