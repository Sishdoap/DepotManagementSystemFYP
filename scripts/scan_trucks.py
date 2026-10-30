"""Scan the depot footage for trucks: when, on which camera, how large.

A feasibility check before building the full truck-pass extractor. For each
video in videos/ it samples frames (default 2 per second), runs a
pretrained COCO YOLOv8n detector for large vehicles, groups detections into
truck passes (src/footage.py), and saves for every pass the frame where the
truck is largest — the best chance of reading its container code.

One worker process per video: frame decoding is the bottleneck and runs on
the CPU, detection runs on the GPU.

Outputs (results/ is gitignored):
    results/truck_scan/detections.csv   every detection (reusable, no re-scan)
    results/truck_scan/passes.csv       one row per pass, with wall-clock times
    results/truck_scan/<camera>/pass_NNN_crop.jpg    full-resolution crop
    results/truck_scan/<camera>/pass_NNN_frame.jpg   whole frame, downscaled

Usage:
    python scripts/scan_trucks.py
    python scripts/scan_trucks.py --sample-fps 1 --max-minutes 5    # quick test

Requires ultralytics (see requirements-ultralytics.txt) and models/yolov8n.pt,
which ultralytics downloads on first use.
"""

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.footage import Detection, group_into_passes, parse_recording_name, wall_clock


# COCO classes a container truck can be detected as. Prime movers with a
# container are sometimes labelled "bus" or "train" by COCO models.
VEHICLE_CLASSES = {5: "bus", 6: "train", 7: "truck"}


def scan_video(args) -> tuple[str, list[Detection], float, float]:
    """Detect large vehicles in sampled frames of one video (worker process)."""
    path, sample_fps, max_minutes, conf, imgsz, model_path = args
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_minutes:
        n_frames = min(n_frames, int(max_minutes * 60 * fps))
    step = max(1, round(fps / sample_fps))
    camera = parse_recording_name(path.name).camera

    detections: list[Detection] = []
    t0 = time.time()
    for idx in range(n_frames):
        if idx % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, imgsz=imgsz, conf=conf, device=0,
                               classes=list(VEHICLE_CLASSES), verbose=False)[0]
        for cls, score, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            detections.append(Detection(
                t=idx / fps, frame=idx, label=VEHICLE_CLASSES[int(cls)],
                conf=float(score), box=tuple(float(v) for v in box),
            ))
        if idx and idx % int(fps * 600) == 0:     # progress every 10 video-minutes
            print(f"  camera {camera}: {idx / fps / 60:.0f}/{n_frames / fps / 60:.0f} min, "
                  f"{len(detections)} detections", flush=True)
    cap.release()
    return camera, detections, fps, time.time() - t0


def save_pass_images(path, passes, out_dir: Path, pad_frac: float = 0.15) -> None:
    """Save the best frame of each pass: a full-resolution crop and a thumbnail."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(passes, 1):
        best = p.best
        cap.set(cv2.CAP_PROP_POS_FRAMES, best.frame)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = best.box
        px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
        crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)),
                     max(0, int(x1 - px)):min(w, int(x2 + px))]
        cv2.imwrite(str(out_dir / f"pass_{i:03d}_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        thumb = frame.copy()
        cv2.rectangle(thumb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
        cv2.imwrite(str(out_dir / f"pass_{i:03d}_frame.jpg"), cv2.resize(thumb, (w // 3, h // 3)))
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--out-dir", default="results/truck_scan")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Detector input width; frames are 5120 px wide and downscaled.")
    parser.add_argument("--max-gap", type=float, default=5.0,
                        help="Seconds without a detection that end a pass.")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Only scan the first N minutes of each video (for testing).")
    args = parser.parse_args()

    videos = sorted(p for p in Path(args.videos_dir).glob("*.mp4") if parse_recording_name(p.name))
    if not videos:
        sys.exit(f"No VLC recordings found in {args.videos_dir}/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_camera = {parse_recording_name(p.name).camera: p for p in videos}
    print(f"Scanning {len(videos)} videos at {args.sample_fps} fps: cameras {', '.join(by_camera)}")

    tasks = [(p, args.sample_fps, args.max_minutes, args.conf, args.imgsz, args.model) for p in videos]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        results = pool.map(scan_video, tasks)

    det_rows, pass_rows = [], []
    for camera, detections, fps, elapsed in sorted(results):
        path = by_camera[camera]
        info = parse_recording_name(path.name)
        passes = group_into_passes(detections, max_gap_seconds=args.max_gap)
        save_pass_images(path, passes, out_dir / camera)
        print(f"camera {camera}: {len(detections)} detections, {len(passes)} passes "
              f"({elapsed / 60:.1f} min)")
        det_rows += [{"camera": camera, "t": round(d.t, 2), "frame": d.frame, "label": d.label,
                      "conf": round(d.conf, 3), "x1": round(d.box[0]), "y1": round(d.box[1]),
                      "x2": round(d.box[2]), "y2": round(d.box[3])} for d in detections]
        for i, p in enumerate(passes, 1):
            b = p.best
            pass_rows.append({
                "camera": camera, "pass": i,
                "start_s": round(p.start, 1), "end_s": round(p.end, 1),
                "duration_s": round(p.end - p.start, 1),
                "wall_start": wall_clock(info, p.start).strftime("%H:%M:%S"),
                "wall_end": wall_clock(info, p.end).strftime("%H:%M:%S"),
                "n_detections": len(p.detections),
                "labels": "/".join(sorted({d.label for d in p.detections})),
                "best_frame": b.frame, "best_conf": round(b.conf, 3),
                "best_w": round(b.box[2] - b.box[0]), "best_h": round(b.box[3] - b.box[1]),
                "crop": f"{camera}/pass_{i:03d}_crop.jpg",
            })

    for name, rows in (("detections.csv", det_rows), ("passes.csv", pass_rows)):
        if rows:
            with (out_dir / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Wrote {out_dir / 'passes.csv'} ({len(pass_rows)} passes) and crops under {out_dir}/")


if __name__ == "__main__":
    main()
