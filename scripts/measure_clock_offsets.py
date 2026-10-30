"""Measure each camera's clock offset and drift from the burned-in clock.

Video time is turned into wall-clock time from the recording start in the
VLC filename (src.footage.wall_clock). That start is when VLC began
recording, not necessarily the camera's own clock, and if the stream
dropped frames, video time would also drift away from real time over the
hour. Cross-camera event assembly needs the cameras on one clock, so this
script measures the difference rather than assuming it is zero.

Each camera burns its time (whole seconds) into the top-left corner. At
several points spread over the recording, the script reads that overlay
on consecutive frames until the displayed second changes; the frame where
it ticks over is when the camera clock reads exactly HH:MM:SS.000. So

    offset = displayed time at the tick - filename start - video time

with a precision of about one frame interval (0.05 s at 20 fps), not one
second. Several points per camera, from the first to the last minute,
show whether the offset drifts (the brief's risk: check the start *and*
end of the hour).

Prints a table and the suggested clock_offset_s for config/cameras.yaml.

Usage:
    python scripts/measure_clock_offsets.py
    python scripts/measure_clock_offsets.py --cameras 558 561 --points 5
"""

import argparse
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.footage import load_camera_configs, parse_overlay_time, parse_recording_name

# Overlay position in full-frame pixels (x1, y1, x2, y2); the same on every camera.
OVERLAY_BOX = (0, 30, 1100, 200)


def read_overlay(paddle, frame) -> str:
    from src.ocr import _flatten_paddle_result

    x1, y1, x2, y2 = OVERLAY_BOX
    crop = frame[y1:y2, x1:x2, ::-1]          # BGR -> RGB
    return " ".join(t for t, _ in _flatten_paddle_result(paddle.predict(crop)))


def measure_point(cap, paddle, info, fps, start_frame, max_frames, step):
    """Find the first frame after `start_frame` where the displayed second changes.

    Returns (video seconds of the tick, offset seconds) or None if the
    overlay could not be read on both sides of the tick.
    """
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev = None                                # (frame index, displayed time)
    for i in range(0, max_frames, step):
        for _ in range(step - 1):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            return None
        idx = start_frame + i + step - 1
        shown = parse_overlay_time(read_overlay(paddle, frame))
        if shown is None:
            continue
        if prev is not None and shown != prev[1]:
            if step > 1 and idx - prev[0] > 1:
                # Coarse pass found the tick between prev[0] and idx; refine frame by frame.
                return measure_point(cap, paddle, info, fps, prev[0] + 1, idx - prev[0], 1)
            t = idx / fps
            displayed = datetime.combine(info.started_at.date(), shown)
            return t, (displayed - info.started_at).total_seconds() - t
        prev = (idx, shown)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--points", type=int, default=8,
                        help="Measurement points spread evenly over each recording.")
    parser.add_argument("--from-s", type=float, default=0.0,
                        help="Only measure from this many seconds into each video.")
    parser.add_argument("--to-s", type=float, default=None,
                        help="Only measure up to this many seconds (to locate a step).")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    import cv2
    from src.ocr import RealOCRAdapter

    paddle = RealOCRAdapter._init_paddle(use_gpu=not args.cpu)
    configs = load_camera_configs(args.config)
    wanted = set(args.cameras) if args.cameras else {c for c, cfg in configs.items()
                                                     if cfg.role != "ignore"}

    for path in sorted(Path(args.videos_dir).glob("*.mp4")):
        info = parse_recording_name(path.name)
        if not info or info.camera not in wanted:
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"camera {info.camera}: {n_frames} frames at {fps} fps "
              f"({n_frames / fps / 60:.1f} min), filename start {info.started_at:%H:%M:%S}")
        offsets = []
        first = int(args.from_s * fps)
        last = min(n_frames, int(args.to_s * fps)) if args.to_s else n_frames
        for k in range(args.points):
            start = first + int((last - first - 3 * fps) * k / max(1, args.points - 1))
            # Coarse steps of 4 frames over 1.5 s always span one tick.
            res = measure_point(cap, paddle, info, fps, start, int(1.5 * fps), 4)
            if res is None:
                print(f"  point {k}: overlay unreadable near {start / fps:.0f} s")
                continue
            t, off = res
            offsets.append((t, off))
            print(f"  video {t:7.2f} s ({t / 60:5.1f} min): offset {off:+.2f} s")
        cap.release()
        if len(offsets) >= 2:
            ts, offs = zip(*offsets)
            slope = statistics.linear_regression(ts, offs).slope * 3600
            print(f"  -> clock_offset_s {statistics.median(offs):+.2f} "
                  f"(range {min(offs):+.2f}..{max(offs):+.2f}, drift {slope:+.2f} s/hour)")


if __name__ == "__main__":
    main()
