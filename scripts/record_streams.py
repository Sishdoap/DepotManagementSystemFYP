"""Record the depot's live camera streams into hourly files (live-stream milestone R1).

One thread per camera (src/recording.py: stream copy, no re-encoding,
Matroska, clock-aligned hourly files, reconnect with back-off). Stream
links come from `.env` (`RTSP_CAM_<n>=rtsp://user:pass@host:<port>/...`,
gitignored); the login is masked in everything this script prints or logs.

Layout (outside the repo by default, so recordings can never be committed):
    ~/depot_recordings/<YYYY-MM-DD>/<HH>00/cam<port>_<YYYY-MM-DD>_<HH-MM-SS>.mkv
    ~/depot_recordings/<YYYY-MM-DD>/recording_log.csv
Each hour folder is a ready input for the pipeline scripts
(`--videos-dir ~/depot_recordings/2026-11-13/0900`).

About 4.5 GB per hour for six cameras. Recording pauses (and logs it) when
free space drops below --min-free-gb; nothing is ever deleted.

Usage:
    python scripts/record_streams.py                         # all cameras, 07:00-19:00
    python scripts/record_streams.py --hours 0 24            # around the clock
    python scripts/record_streams.py --cameras 558 561 --root /tmp/rec
Stop with Ctrl+C or SIGTERM (files are closed cleanly).
"""

import argparse
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.recording import RecordingLog, StreamRecorder, in_hours, load_stream_urls, mask_url


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=str(root_dir / ".env"))
    parser.add_argument("--root", default=str(Path.home() / "depot_recordings"))
    parser.add_argument("--cameras", nargs="*", default=None, help="Ports, e.g. 557 561 (default: all in .env)")
    parser.add_argument("--hours", nargs=2, type=int, default=(7, 19), metavar=("START", "END"),
                        help="Record from START:00 to END:00 local time (default 7 19)")
    parser.add_argument("--min-free-gb", type=float, default=50.0)
    parser.add_argument("--segment-s", type=int, default=3600)
    parser.add_argument("--status-every-s", type=float, default=300)
    args = parser.parse_args()

    urls = load_stream_urls(args.env)
    if args.cameras:
        missing = set(args.cameras) - set(urls)
        if missing:
            sys.exit(f"Not in {args.env}: {', '.join(sorted(missing))}")
        urls = {c: urls[c] for c in args.cameras}
    if not urls:
        sys.exit(f"No RTSP_CAM_* links in {args.env}")

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    log = RecordingLog(root)
    start_h, end_h = args.hours
    low_disk = threading.Event()

    def should_continue(now):
        return in_hours(now, start_h, end_h) and not low_disk.is_set()

    recorders = [StreamRecorder(cam, url, root, log=log, segment_s=args.segment_s)
                 for cam, url in sorted(urls.items())]
    threads = [threading.Thread(target=r.run, kwargs={"should_continue": should_continue},
                                name=f"cam{r.camera}", daemon=True) for r in recorders]

    stopping = threading.Event()

    def stop(*_):
        stopping.set()
        for r in recorders:
            r.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"Recording cameras {', '.join(r.camera for r in recorders)} to {root}, "
          f"{start_h:02d}:00-{end_h:02d}:00, segments of {args.segment_s} s")
    for r in recorders:
        print(f"  {r.camera}: {mask_url(r.source)}")
    for t in threads:
        t.start()

    last_status = 0.0
    while not stopping.is_set():
        free_gb = shutil.disk_usage(root).free / 1e9
        if free_gb < args.min_free_gb and not low_disk.is_set():
            low_disk.set()
            log.write("all", "paused", f"free space {free_gb:.0f} GB < {args.min_free_gb:.0f} GB")
            print(f"PAUSED: only {free_gb:.0f} GB free", flush=True)
        elif free_gb >= args.min_free_gb and low_disk.is_set():
            low_disk.clear()
            log.write("all", "resumed", f"free space {free_gb:.0f} GB")
        if time.time() - last_status >= args.status_every_s:
            last_status = time.time()
            state = "recording" if should_continue(datetime.now()) else "waiting"
            parts = [f"{r.camera}: {len(r.segments)} files"
                     + (f", {r.segments[-1].packets} pkts in {r.segments[-1].path.name}" if r.segments else "")
                     for r in recorders]
            print(f"[{datetime.now():%H:%M:%S}] {state}, {free_gb:.0f} GB free | " + " | ".join(parts),
                  flush=True)
        stopping.wait(5)

    for t in threads:
        t.join(timeout=20)
    print("Stopped.")


if __name__ == "__main__":
    main()
