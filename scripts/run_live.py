"""Live truck identification: recordings in, vehicle profiles in the database out.

Follows the hourly files that scripts/record_streams.py is writing (no
extra camera connections; src/live_source.py), and for every truck that
passes the checkpoint writes one row to `truck_passes` (src/db.py):
container code(s), size/type, vehicle type, vehicle and trailer plate,
capacity class — first as "provisional" while the truck is in view, then
"final" once it has left and its crops have been read.

    decoder thread per camera ──► main loop ──────────────────► database
      (src.live_source)            detect + track (one YOLO, src.live_tracking)
                                   vehicle type per detection
                                   crops for OCR ─► spool dir
                                   event assembly (src.live_events)
                                   PassRecorder (src.live_pipeline)
                                        │ final event          ▲ reads
                                        ▼                      │
                                   OCR process (PaddleOCR, src.live_ocr)

PaddleOCR runs in its own process with the footage memory caps (detection
side 1920 px, 6 GB GPU memory; see src/ocr.py); YOLO and the type
classifier share one CUDA context in the main process.

With --replay <hour folder> the same pipeline runs over a finished hour
as fast as the machine allows (demo fallback, and the check against the
batch results). Live, it starts at the beginning of the file being
written now, ~30 s before the present (--from-hour-start: from the
beginning of that file, catching up at ~3.4x real time).

Clock offsets: live recordings are stamped with the PC clock, measured
within 1 s of the camera overlay (13 Nov), so live mode uses zero offsets
unless --offsets-from points at an hour folder with a measured
clock_offsets.yaml. --replay uses that hour's clock_offsets.yaml when present.

The database file (default depot_live.db) holds plates and stays out of git.

Usage:
    python scripts/run_live.py                                   # live
    python scripts/run_live.py --replay ~/depot_recordings/2026-11-13/1100 \
        --db sqlite:///results/recordings/2026-11-13_1100/live_replay.db
"""

import argparse
import dataclasses
import multiprocessing as mp
import queue
import signal
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.footage import find_recordings, load_camera_configs  # noqa: E402

DONE = object()


def ocr_worker(jobs, results, roles, frames_per_camera, det_max_side, gpu_mem_mb, use_gpu):
    """OCR process: read each final event's crops, send the reads back."""
    import os
    # Ctrl-C / SIGTERM reach the whole process group; the main process ends
    # this one by sending None after the last job.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sys.path.insert(0, str(root_dir))
    from src.live_ocr import EventReads, PaddleTextReader, read_event
    reader = PaddleTextReader(use_gpu=use_gpu, det_max_side=det_max_side, gpu_mem_mb=gpu_mem_mb)
    results.put(("ready", None, None))
    while True:
        job = jobs.get()
        if job is None:
            break
        t0 = time.time()
        try:
            reads = read_event(reader, job.crops, roles, max_frames_per_camera=frames_per_camera)
            err = None
        except Exception as e:                       # one bad event must not stop the pipeline
            reads, err = EventReads(), f"{type(e).__name__}: {e}"
        results.put((job.pass_uid, reads, err or f"{time.time() - t0:.0f} s"))
        for c in job.crops:
            try:
                os.remove(c.path)
            except OSError:
                pass


def decode(follower, q, stop):
    try:
        for sf in follower.frames():
            while not stop.is_set():
                try:
                    q.put(sf, timeout=0.5)
                    break
                except queue.Full:
                    continue
            if stop.is_set():
                break
    finally:
        q.put(DONE)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path.home() / "depot_recordings"),
                        help="Recorder output folder (live mode).")
    parser.add_argument("--replay", default=None, help="Replay this finished hour folder instead.")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--offsets-from", default=None,
                        help="Hour folder whose clock_offsets.yaml to use in live mode.")
    parser.add_argument("--db", default="sqlite:///depot_live.db")
    parser.add_argument("--spool", default=str(Path.home() / "depot_live_spool"))
    parser.add_argument("--model", default="models/yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--sample-every", type=int, default=5, help="Every Nth frame (20 fps / 5 = 4 fps).")
    parser.add_argument("--frames-per-camera", type=int, default=60,
                        help="OCR frame budget per camera per truck; 0 = all.")
    parser.add_argument("--det-max-side", type=int, default=1920)
    parser.add_argument("--gpu-mem-mb", type=int, default=6144)
    parser.add_argument("--cpu-ocr", action="store_true")
    parser.add_argument("--update-every-s", type=float, default=5.0)
    parser.add_argument("--status-every-s", type=float, default=60.0)
    parser.add_argument("--from-hour-start", action="store_true",
                        help="Live: read the current hour from its start (default: from ~30 s ago).")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Replay: stop after this many minutes of video (testing).")
    args = parser.parse_args()

    from src.db import init_schema, list_passes, make_engine
    from src.live_events import LiveEventAssembler
    from src.live_pipeline import PassRecorder
    from src.live_source import CameraFollower, camera_files
    from src.live_tracking import (CropSpool, MultiCameraTracker, UltralyticsDetector, frame_types,
                                   type_crops)
    from src.vehicle_type import UltralyticsVehicleTypeClassifier

    replay = args.replay is not None
    if replay:
        hour = Path(args.replay).expanduser()
        configs = load_camera_configs(args.config, videos_dir=hour)
        files = find_recordings(hour)
    else:
        configs = load_camera_configs(args.config, videos_dir=args.offsets_from)
        if not args.offsets_from:
            configs = {c: dataclasses.replace(cfg, clock_offsets=((0.0, 0.0),))
                       for c, cfg in configs.items()}
        files = {}
    cameras = [c for c, cfg in configs.items() if cfg.role != "ignore"
               and (not replay or c in files)]
    roles = {c: configs[c].role for c in configs}
    print(f"{'Replaying ' + str(hour) if replay else 'Live from ' + args.root}: cameras "
          f"{', '.join(cameras)}", flush=True)

    engine = make_engine(args.db)
    init_schema(engine)
    spool = CropSpool(Path(args.spool).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S"), roles)

    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    ocr = ctx.Process(target=ocr_worker, daemon=True, args=(
        jobs, results, roles, args.frames_per_camera or None, args.det_max_side or None,
        args.gpu_mem_mb or None, not args.cpu_ocr))
    ocr.start()
    kind, _, _ = results.get()                        # wait for PaddleOCR before taking the GPU
    assert kind == "ready"

    tracker = MultiCameraTracker(UltralyticsDetector(args.model, args.imgsz, args.conf))
    classifier = UltralyticsVehicleTypeClassifier()
    recorder = PassRecorder(engine, jobs.put, source="replay" if replay else "live")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    followers, queues, threads = {}, {}, []
    for cam in cameras:
        skip = 0.0
        if not replay and not args.from_hour_start:
            current = camera_files(args.root, cam)
            if current:
                skip = max(0.0, (datetime.now() - current[-1][0].started_at).total_seconds() - 30)
        f = CameraFollower(args.root, cam, live=not replay, sample_every=args.sample_every,
                           files=[files[cam]] if replay else None, skip_s=skip)
        followers[cam], queues[cam] = f, queue.Queue(maxsize=4)
        th = threading.Thread(target=decode, args=(f, queues[cam], stop), daemon=True)
        th.start()
        threads.append(th)

    # The assembler needs each camera's reference recording, known once its follower has opened it.
    assembler = None
    open_cams = set(cameras)
    latest_t: dict[str, float] = {}
    frames_done, ocr_done, t_start = Counter(), 0, time.time()
    next_update = next_status = 0.0
    pending_frames: dict[str, list] = {c: [] for c in cameras}

    def handle_results(block=False):
        nonlocal ocr_done
        while True:
            try:
                uid, reads, info = results.get(timeout=1.0 if block else 0.0)
            except queue.Empty:
                return
            p = recorder.on_reads(uid, reads)
            ocr_done += 1
            print(f"  final {p.entry_time:%H:%M:%S} {' '.join(p.codes) or '-':24} "
                  f"{p.size_type or '-':5} {p.vehicle_type:17} {p.capacity_class:7} "
                  f"plates {'yes' if p.plate else '-'}/{'yes' if p.trailer_plate else '-'} "
                  f"({reads.frames_read} frames, {info})", flush=True)
            if not block:
                continue
            return

    while open_cams and not stop.is_set():
        batch = []
        for cam in list(open_cams):
            try:
                sf = queues[cam].get(timeout=0.02)
            except queue.Empty:
                continue
            if sf is DONE:
                open_cams.discard(cam)
                continue
            if args.max_minutes and sf.t > args.max_minutes * 60:
                followers[cam].stop()
                continue
            batch.append(sf)
        for cam, f in followers.items():
            if assembler is not None:
                assembler.set_stalled(cam, f.stalled)
        if not batch:
            handle_results()
            continue
        if assembler is None and all(followers[c].reference for c in cameras):
            assembler = LiveEventAssembler(configs, {c: followers[c].reference for c in cameras})
        out = tracker.step(batch)
        for sf in batch:
            dets = out.get(sf.camera, [])
            latest_t[sf.camera] = sf.t
            frames_done[sf.camera] += 1
            if roles[sf.camera] != "ignore" and dets:
                keep, crops = type_crops(sf.image, dets)
                recorder.add_types(sf.camera, frame_types(classifier, sf.camera, keep, crops))
                for d in dets:
                    c = spool.save(sf.camera, d, sf.image)
                    if c is not None:
                        recorder.add_crop(c)
            if assembler is None:
                pending_frames[sf.camera].append((sf.t, dets))
            else:
                for t, ds in pending_frames.pop(sf.camera, []):
                    assembler.add_frame(sf.camera, t, ds)
                assembler.add_frame(sf.camera, sf.t, dets)
        if assembler is not None:
            now = assembler.now()
            if now is not None and now.timestamp() >= next_update:
                next_update = now.timestamp() + args.update_every_s
                recorder.on_update(assembler.update())
                spool.remove(recorder.drop_crops_before(latest_t, keep_s=900))
        handle_results()
        if time.time() >= next_status:
            next_status = time.time() + args.status_every_s
            now = assembler.now() if assembler else None
            lag = f", {(datetime.now() - now).total_seconds():.0f} s behind" if now and not replay else ""
            print(f"[{datetime.now():%H:%M:%S}] video {now:%H:%M:%S}{lag}; " if now else
                  f"[{datetime.now():%H:%M:%S}] starting; ", end="")
            print(f"frames {dict(frames_done)}; OCR pending {len(recorder.pending)}, "
                  f"done {ocr_done}", flush=True)

    stop.set()
    for f in followers.values():
        f.stop()
    if replay and assembler is not None:
        recorder.on_update(assembler.update(final_all=True))
        print(f"Video done in {time.time() - t_start:.0f} s; waiting for OCR of "
              f"{len(recorder.pending)} trucks", flush=True)
        while recorder.pending:
            handle_results(block=True)
    jobs.put(None)
    ocr.join(timeout=60)
    final = list_passes(engine, status="final", limit=10000)
    print(f"Done in {time.time() - t_start:.0f} s: {len(final)} final passes in {args.db}")


if __name__ == "__main__":
    main()
