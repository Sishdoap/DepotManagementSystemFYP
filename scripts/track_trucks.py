"""Track trucks through the depot footage with YOLO + ByteTrack.

First stage of the event extractor (milestone M1). scripts/scan_trucks.py
grouped detections into passes by time gaps alone, so two trucks driving
back to back merged into one pass, and parked or passing vehicles outside
the entry lane became passes of their own. This script instead runs a
tracker, so every detection carries a track id that follows one vehicle
from frame to frame.

It only detects and tracks; lane masks, size limits and cross-camera
assembly are applied afterwards by scripts/extract_events.py from
config/cameras.yaml. Keeping the expensive scan separate means the masks
can be tuned without decoding an hour of 5120x1440 video again.

The confidence threshold is deliberately low (0.1): ByteTrack's second
association step uses low-score boxes to keep a track alive through
partial occlusion, and the stored `conf` lets later stages filter.

Only cameras whose role in config/cameras.yaml is not "ignore" are
scanned. One worker process per video: decoding runs on the CPU and is the
bottleneck, detection runs on the GPU.

Output (results/ is gitignored):
    results/events/tracks.csv   one row per tracked detection

Usage:
    python scripts/track_trucks.py
    python scripts/track_trucks.py --cameras 558 --max-minutes 5    # quick test
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

from src.footage import Detection, load_camera_configs, parse_recording_name


# COCO classes a container truck can be detected as. Prime movers with a
# container are sometimes labelled "bus" or "train" by COCO models.
VEHICLE_CLASSES = {5: "bus", 6: "train", 7: "truck"}


def track_video(args) -> tuple[str, list[Detection], float]:
    """Detect and track large vehicles in sampled frames of one video (worker process)."""
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
        result = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=imgsz,
                             conf=conf, device=0, classes=list(VEHICLE_CLASSES),
                             verbose=False)[0]
        if result.boxes.id is not None:
            for cls, score, box, tid in zip(result.boxes.cls, result.boxes.conf,
                                            result.boxes.xyxy, result.boxes.id):
                detections.append(Detection(
                    t=idx / fps, frame=idx, label=VEHICLE_CLASSES[int(cls)],
                    conf=float(score), box=tuple(float(v) for v in box), track_id=int(tid),
                ))
        if idx and idx % int(fps * 600) == 0:     # progress every 10 video-minutes
            print(f"  camera {camera}: {idx / fps / 60:.0f}/{n_frames / fps / 60:.0f} min, "
                  f"{len(detections)} tracked detections", flush=True)
    cap.release()
    return camera, detections, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--out", default="results/events/tracks.csv")
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Camera ids to scan (default: every camera not marked 'ignore').")
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Detector input width; frames are 5120 px wide and downscaled.")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Only scan the first N minutes of each video (for testing).")
    args = parser.parse_args()

    configs = load_camera_configs(args.config)
    wanted = set(args.cameras) if args.cameras else {c for c, cfg in configs.items()
                                                     if cfg.role != "ignore"}
    videos = sorted(p for p in Path(args.videos_dir).glob("*.mp4")
                    if (info := parse_recording_name(p.name)) and info.camera in wanted)
    if not videos:
        sys.exit(f"No recordings for cameras {sorted(wanted)} in {args.videos_dir}/")
    print(f"Tracking {len(videos)} videos at {args.sample_fps} fps: cameras "
          f"{', '.join(parse_recording_name(p.name).camera for p in videos)}")

    tasks = [(p, args.sample_fps, args.max_minutes, args.conf, args.imgsz, args.model)
             for p in videos]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        results = pool.map(track_video, tasks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera", "track_id", "t", "frame", "label", "conf",
                         "x1", "y1", "x2", "y2"])
        for camera, detections, elapsed in sorted(results):
            print(f"camera {camera}: {len(detections)} tracked detections, "
                  f"{len({d.track_id for d in detections})} tracks ({elapsed / 60:.1f} min)")
            for d in detections:
                writer.writerow([camera, d.track_id, round(d.t, 2), d.frame, d.label,
                                 round(d.conf, 3), *(round(v) for v in d.box)])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
