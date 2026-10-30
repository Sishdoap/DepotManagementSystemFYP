"""Turn tracked footage into checkpoint events (milestone M1).

Second stage of the event extractor, after scripts/track_trucks.py:

1. read results/events/tracks.csv (one row per tracked detection);
2. per camera, keep only tracks of trucks in the entry lane
   (src.footage.keep_entry_lane_passes, settings in config/cameras.yaml);
3. convert video time to wall-clock time with each camera's measured
   clock offset, so all cameras share one clock;
4. assemble the kept passes into events, one per truck through the
   checkpoint (src.events.assemble_events; chain order = camera roles
   cab -> codes -> backup);
5. write events.csv, a contact sheet per event for checking by eye, and
   rejected_tracks.csv listing every dropped track and why.

No OCR happens here; reading codes per frame is milestone M2.

Outputs (results/ is gitignored):
    results/events/events.csv
    results/events/rejected_tracks.csv
    results/events/sheets/event_NNN.jpg

Usage:
    python scripts/extract_events.py
    python scripts/extract_events.py --no-sheets      # CSVs only, no video decoding
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

from src.events import extract_events
from src.footage import Detection, load_camera_configs, load_tracks_csv, parse_recording_name

SHEET_FRAMES_PER_CAMERA = 4
SHEET_TILE_HEIGHT = 360


def sample_detections(dets: list[Detection], n: int) -> list[Detection]:
    """n detections spread evenly over a pass, always including the largest box."""
    if len(dets) <= n:
        return dets
    picks = {round(i * (len(dets) - 1) / (n - 1)) for i in range(n)}
    best = max(range(len(dets)), key=lambda i: dets[i].area)
    if best not in picks:
        picks.remove(min(picks, key=lambda i: abs(i - best)))
        picks.add(best)
    return [dets[i] for i in sorted(picks)]


def write_contact_sheets(events, track_dets, videos, chain, out_dir: Path) -> list[str]:
    """One image per event: a row of crops per camera, frames in time order."""
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    # Collect every frame needed, grouped per camera, so each video is opened once.
    wanted: dict[str, set[int]] = defaultdict(set)
    plan = []                                 # per event: [(camera, [Detection, ...]), ...]
    for e in events:
        rows = []
        for cam in chain:
            dets = [d for p in e.on(cam) for d in track_dets[(cam, p.track_id)]]
            if dets:
                picked = sample_detections(sorted(dets, key=lambda d: d.t), SHEET_FRAMES_PER_CAMERA)
                rows.append((cam, picked))
                wanted[cam].update(d.frame for d in picked)
        plan.append(rows)

    frames: dict[tuple[str, int], "np.ndarray"] = {}
    for cam, idxs in wanted.items():
        cap = cv2.VideoCapture(str(videos[cam]))
        for idx in sorted(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames[(cam, idx)] = frame
        cap.release()

    paths = []
    for i, (e, rows) in enumerate(zip(events, plan), 1):
        strips = []
        for cam, picked in rows:
            tiles = []
            for d in picked:
                frame = frames.get((cam, d.frame))
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = d.box
                px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)),
                             max(0, int(x1 - px)):min(w, int(x2 + px))]
                scale = SHEET_TILE_HEIGHT / crop.shape[0]
                tile = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), SHEET_TILE_HEIGHT))
                cv2.putText(tile, f"{cam} #{d.track_id} t={d.t:.1f}s", (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                tiles.append(tile)
            if tiles:
                strips.append(np.hstack(tiles))
        if not strips:
            continue
        width = max(s.shape[1] for s in strips)
        header = np.zeros((40, width, 3), np.uint8)
        cv2.putText(header, f"event {i:03d}  {e.start:%H:%M:%S}-{e.end:%H:%M:%S}  "
                    f"cameras {'+'.join(e.cameras)}{'' if e.head_seen else '  (head camera missed)'}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        sheet = np.vstack([header] + [np.pad(s, ((0, 0), (0, width - s.shape[1]), (0, 0)))
                                      for s in strips])
        path = out_dir / f"event_{i:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path.name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--tracks", default="results/events/tracks.csv")
    parser.add_argument("--out-dir", default="results/events")
    parser.add_argument("--max-lag", type=float, default=45.0,
                        help="Latest (s) a later camera may first see a truck after the event starts.")
    parser.add_argument("--merge-gap", type=float, default=25.0,
                        help="Largest gap (s) between fragments of one truck on one camera.")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    configs = load_camera_configs(args.config)
    videos = {info.camera: p for p in Path(args.videos_dir).glob("*.mp4")
              if (info := parse_recording_name(p.name))}
    infos = {cam: parse_recording_name(p.name) for cam, p in videos.items()}
    out_dir = Path(args.out_dir)

    ex = extract_events(load_tracks_csv(args.tracks), configs, infos,
                        max_lag_s=args.max_lag, merge_gap_s=args.merge_gap)
    chain, events, track_dets, rejected = ex.chain, ex.events, ex.detections, ex.rejected
    for cam in chain:
        n_kept = sum(1 for c, _ in track_dets if c == cam)
        n_rej = sum(1 for r in rejected if r["camera"] == cam)
        print(f"camera {cam} ({configs[cam].role}): {n_kept + n_rej} tracks, {n_kept} kept")

    sheets = ([] if args.no_sheets else
              write_contact_sheets(events, track_dets, videos, chain, out_dir / "sheets"))

    rows = []
    for i, e in enumerate(events, 1):
        row = {"event": i, "entry_time": f"{e.start:%Y-%m-%d %H:%M:%S}",
               "end_time": f"{e.end:%H:%M:%S}",
               "duration_s": round((e.end - e.start).total_seconds(), 1),
               "cameras": "+".join(e.cameras), "head_seen": e.head_seen}
        for cam in chain:
            on = e.on(cam)
            row[f"cam{cam}_start"] = f"{on[0].start:%H:%M:%S}" if on else ""
            row[f"cam{cam}_end"] = f"{on[-1].end:%H:%M:%S}" if on else ""
            row[f"cam{cam}_tracks"] = " ".join(str(p.track_id) for p in on)
            best = max((d for p in on for d in track_dets[(cam, p.track_id)]),
                       key=lambda d: d.area, default=None)
            row[f"cam{cam}_best_frame"] = best.frame if best else ""
        row["contact_sheet"] = f"sheets/event_{i:03d}.jpg" if f"event_{i:03d}.jpg" in sheets else ""
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("events.csv", rows), ("rejected_tracks.csv", rejected)):
        if data:
            with (out_dir / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    missed = sum(not e.head_seen for e in events)
    print(f"{len(events)} events ({missed} not seen by head camera {chain[0]}); "
          f"{len(rejected)} tracks rejected")
    for r in rows:
        print(f"  {r['event']:3d}  {r['entry_time'][11:]}  {r['duration_s']:5.1f} s  {r['cameras']}")
    print(f"Wrote {out_dir / 'events.csv'}" + (f" and {len(sheets)} contact sheets" if sheets else ""))


if __name__ == "__main__":
    main()
