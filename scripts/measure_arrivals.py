"""Measure truck arrivals and checkpoint times from replayed recordings (FR7).

Reads the databases that `scripts/run_live.py --replay <day folders>`
wrote and the recordings themselves (for how much of each hour was
recorded), and reports, using src/arrivals.py:

- trucks per hour of the day and per minute, day (07-19) vs night;
- the capacity-class and vehicle-type mix;
- whether arrivals look Poisson (inter-arrival CV, counts per 10 min);
- time through the checkpoint (first to last camera), waiting before the
  barrier camera (561) and time in view of it, per capacity class.

Outputs (in --out, gitignored with the rest of results/): hours.csv,
passes.csv (times, type, class, number of codes; no plates or codes) and
summary.json for the routing evaluation (M9).

Usage:
    python scripts/measure_arrivals.py \
        --db results/arrivals/2026-11-13.db results/arrivals/2026-11-14.db \
        --recordings ~/depot_recordings/2026-11-13 ~/depot_recordings/2026-11-14
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.arrivals import (by_group, cv, dispersion_index, hour_start, hourly_counts,  # noqa: E402
                          interarrivals, is_day, pass_timings, rate_per_minute, summarise)
from src.db import list_pass_tracks, list_passes, make_engine  # noqa: E402
from src.live_source import replay_files  # noqa: E402
from src.footage import parse_recording_name  # noqa: E402

FPS = 20.0


def logged_packets(day_dir: Path) -> dict[str, int]:
    """{file name: packets} from the recorder's segment_closed lines."""
    log = day_dir / "recording_log.csv"
    out = {}
    if log.exists():
        with open(log) as f:
            for r in csv.DictReader(f):
                if r.get("event") == "segment_closed" and "packets=" in (r.get("detail") or ""):
                    name, rest = r["detail"].split(" ", 1)
                    out[name] = int(rest.split("packets=")[1].split()[0])
    return out


def probe_packets(path: Path) -> int:
    """Video packets in a file (demux only, no decoding); for files the log has no count for."""
    import av
    n = 0
    try:
        with av.open(str(path)) as c:
            for pkt in c.demux(c.streams.video[0]):
                if pkt.size:
                    n += 1
    except (av.error.InvalidDataError, av.error.EOFError):
        pass                                   # truncated: count what was readable
    return n


def coverage(folders, camera: str) -> dict[datetime, float]:
    """{clock hour: seconds recorded} from `camera`'s files under the folders."""
    files = replay_files(folders).get(camera, [])
    logs: dict[Path, dict[str, int]] = {}
    cov: dict[datetime, float] = {}
    for p in files:
        day = next((a for a in p.parents if (a / "recording_log.csv").exists()), p.parent)
        if day not in logs:
            logs[day] = logged_packets(day)
        n = logs[day].get(p.name)
        if n is None:
            n = probe_packets(p)
        h = hour_start(parse_recording_name(p.name).started_at)
        cov[h] = min(3600.0, cov.get(h, 0.0) + n / FPS)
    return cov


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", nargs="+", required=True, help="Replay databases (sqlite files or URLs).")
    ap.add_argument("--recordings", nargs="+", required=True,
                    help="The hour/day folders that were replayed (for recorded time per hour).")
    ap.add_argument("--barrier-camera", default="561")
    ap.add_argument("--out", default="results/arrivals")
    args = ap.parse_args()

    passes, tracks = [], {}
    for db in args.db:
        engine = make_engine(db if "://" in db else f"sqlite:///{db}")
        passes += list_passes(engine, status="final", limit=1_000_000)
        tracks.update(list_pass_tracks(engine))
        engine.dispose()
    timings = pass_timings(passes, tracks, barrier_camera=args.barrier_camera)
    cov = coverage([Path(r).expanduser() for r in args.recordings], args.barrier_camera)
    rows = hourly_counts(timings, cov)
    skipped = len([p for p in passes if p["pass_uid"] not in tracks])
    print(f"{len(timings)} trucks over {sum(cov.values()) / 3600:.1f} recorded hours "
          f"({len(cov)} clock hours){f'; {skipped} passes without tracks skipped' if skipped else ''}")

    day_rows = [r for r in rows if is_day(r.hour)]
    night_rows = [r for r in rows if not is_day(r.hour)]
    day_t = [t for t in timings if is_day(t.entry)]
    night_t = [t for t in timings if not is_day(t.entry)]
    day_cov = {r.hour: r.recorded_s for r in day_rows}
    gaps_all, gaps_day = interarrivals(timings, cov), interarrivals(day_t, day_cov)

    print("\n--- Trucks per hour (by checkpoint time) ---")
    print(f"{'hour':16} {'rec min':>7} {'trucks':>6} {'/min':>5}  classes")
    for r in rows:
        mix = " ".join(f"{k}={v}" for k, v in sorted(r.by_class.items()))
        print(f"{r.hour:%a %d %b %H:00} {r.recorded_s / 60:7.0f} {r.n:6d} {fmt(r.per_minute, 2):>5}  {mix}")

    print("\n--- Rates ---")
    for name, rs in (("all", rows), ("day 07-19", day_rows), ("night", night_rows)):
        print(f"{name:10} {fmt(rate_per_minute(rs), 3)} trucks/min "
              f"({sum(r.n for r in rs)} trucks, {sum(r.recorded_s for r in rs) / 3600:.1f} h)")
    busiest = max(rows, key=lambda r: r.per_minute or 0, default=None)
    if busiest:
        print(f"busiest hour {busiest.hour:%a %d %b %H:00}: {busiest.n} trucks "
              f"({fmt(busiest.per_minute, 2)}/min)")
    hod = Counter()
    hod_s = Counter()
    for r in rows:
        hod[r.hour.hour] += r.n
        hod_s[r.hour.hour] += r.recorded_s
    print("by hour of day (/min): " + "  ".join(
        f"{h:02d}:{hod[h] / (hod_s[h] / 60):.2f}" for h in sorted(hod_s) if hod_s[h] > 0))

    print("\n--- Is it Poisson? (CV of gaps ~1, dispersion of 10-min counts ~1) ---")
    print(f"all hours: gap CV {fmt(cv(gaps_all), 2)} (n={len(gaps_all)}), "
          f"dispersion {fmt(dispersion_index(timings, cov), 2)}")
    print(f"day hours: gap CV {fmt(cv(gaps_day), 2)} (n={len(gaps_day)}), "
          f"dispersion {fmt(dispersion_index(day_t, day_cov), 2)}")

    print("\n--- Mix ---")
    for name, key in (("capacity class", lambda t: t.capacity_class),
                      ("vehicle type", lambda t: t.vehicle_type)):
        c = Counter(key(t) for t in timings)
        print(f"{name}: " + ", ".join(f"{k} {v} ({v / len(timings):.0%})" for k, v in c.most_common()))

    print(f"\n--- Seconds at the checkpoint, per class: through (first to last camera), "
          f"waiting before {args.barrier_camera}, in view of {args.barrier_camera} ---")
    print(f"{'class':10} {'n':>4} {'through median':>14} {'p90':>6}   {'wait median':>11} {'p90':>6}   "
          f"{'barrier median':>14} {'p90':>6}")
    through = by_group(timings, lambda t: t.through_s)
    wait = by_group(timings, lambda t: t.wait_s)
    barrier = by_group(timings, lambda t: t.barrier_s)
    empty = summarise([])
    for g in [*sorted(through), "all"]:
        if g == "all":
            th, w, b = (summarise([f(t) for t in timings]) for f in
                        (lambda t: t.through_s, lambda t: t.wait_s, lambda t: t.barrier_s))
        else:
            th, w, b = through[g], wait.get(g, empty), barrier.get(g, empty)
        print(f"{g:10} {th.n:4d} {fmt(th.median):>14} {fmt(th.p90):>6}   {fmt(w.median):>11} "
              f"{fmt(w.p90):>6}   {fmt(b.median):>14} {fmt(b.p90):>6}")
    b_all = summarise([t.barrier_s for t in timings])
    print(f"no barrier-camera track: {sum(t.barrier_s is None for t in timings)} trucks")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hours.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "recorded_s", "trucks", "per_minute", "classes"])
        for r in rows:
            w.writerow([r.hour.isoformat(), round(r.recorded_s), r.n, fmt(r.per_minute, 4),
                        " ".join(f"{k}={v}" for k, v in sorted(r.by_class.items()))])
    with open(out / "passes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pass_uid", "first_seen", "entry", "end", "barrier_s", "wait_s", "through_s",
                    "vehicle_type", "capacity_class", "n_codes"])
        for t in timings:
            w.writerow([t.pass_uid, t.first_seen.isoformat(), t.entry.isoformat(), t.end.isoformat(),
                        fmt(t.barrier_s), fmt(t.wait_s), fmt(t.through_s), t.vehicle_type,
                        t.capacity_class, t.n_codes])

    def s(x):
        return x.__dict__
    summary = {
        "databases": args.db, "recordings": args.recordings,
        "created": datetime.now().isoformat(timespec="seconds"),
        "trucks": len(timings), "recorded_hours": sum(cov.values()) / 3600,
        "rate_per_minute": {"all": rate_per_minute(rows), "day": rate_per_minute(day_rows),
                            "night": rate_per_minute(night_rows),
                            "busiest_hour": busiest.per_minute if busiest else None},
        "poisson": {"gap_cv_all": cv(gaps_all), "gap_cv_day": cv(gaps_day),
                    "dispersion_all": dispersion_index(timings, cov),
                    "dispersion_day": dispersion_index(day_t, day_cov)},
        "capacity_class_mix": dict(Counter(t.capacity_class for t in timings)),
        "vehicle_type_mix": dict(Counter(t.vehicle_type for t in timings)),
        "through_s": {g: s(v) for g, v in through.items()},
        "through_s_all": s(summarise([t.through_s for t in timings])),
        "wait_s": {g: s(v) for g, v in wait.items()},
        "barrier_s": {g: s(v) for g, v in barrier.items()},
        "barrier_s_all": s(b_all),
        "day_trucks": len(day_t), "night_trucks": len(night_t),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}/hours.csv, passes.csv, summary.json")


if __name__ == "__main__":
    main()
