"""Truck arrivals and checkpoint times measured from the recordings (FR7).

The gate simulation (src/simulation.py) draws Poisson arrivals at a fixed
rate and gate service times from assumed distributions. Under the
2026-10-30 decision ("option B") the depot cameras are the *entry
identification checkpoint* and gates A/B/C stay simulated, so what the
recordings can give the simulation is:

- **Arrivals:** when each truck reaches the checkpoint, hence the rate per
  hour of the day, day vs night, and whether arrivals look Poisson
  (inter-arrival coefficient of variation near 1, counts per 10 minutes
  with variance near their mean).
- **The vehicle mix:** vehicle type and capacity class (src/profiles.py)
  of each arrival — what a capacity-class service time would be drawn
  for.
- **Time through the checkpoint** (`through_s`, first to last detection
  on any camera) per capacity class, and its parts: waiting before the
  barrier camera (`wait_s`; on 12 Aug the side cameras saw a truck wait
  up to ~90 s) and time in view of the barrier camera (561, `barrier_s`).
  These are observations of the camera checkpoint, not gate service times
  (the gates are simulated). On the 13 Nov 11:00 check hour trucks drove
  through in ~10-25 s and 561 saw each cab for only 1-3 s: no stop there
  to time, so `barrier_s` is a sanity check, not a service-time estimate.

Input is what scripts/run_live.py stores (src/db.py): final truck passes
and their camera tracks (`pass_tracks`). Rates are counts divided by the
time actually recorded (`coverage`), so a lost hour or a part hour does
not read as a quiet one.

Every number here is an observation of this depot over the recorded days,
from an automatic pipeline (missed or split trucks included); the report
states both.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Optional, Sequence

DAY_HOURS = range(7, 19)            # 07:00-18:59 is "day", the rest "night"


@dataclass(frozen=True)
class PassTiming:
    """One truck through the checkpoint, with the times FR7 needs."""
    pass_uid: str
    first_seen: datetime            # first detection on any camera (arrival, waiting included)
    entry: datetime                 # checkpoint time (the barrier camera's pass)
    end: datetime                   # last detection on any camera
    barrier_s: Optional[float]      # time in view on the barrier camera; None if not seen there
    vehicle_type: str
    capacity_class: str
    n_codes: int

    @property
    def wait_s(self) -> float:
        """Queueing before the barrier: first seen anywhere -> barrier camera."""
        return max(0.0, (self.entry - self.first_seen).total_seconds())

    @property
    def through_s(self) -> float:
        """First seen -> last seen: the whole time at the checkpoint."""
        return (self.end - self.first_seen).total_seconds()


def pass_timings(passes: Iterable[Mapping], tracks: Mapping[str, Sequence[Mapping]],
                 barrier_camera: str = "561") -> list[PassTiming]:
    """PassTiming per final pass, oldest first.

    `passes` are rows of src.db.list_passes, `tracks` is
    src.db.list_pass_tracks. The barrier time spans all of the pass's
    tracks on `barrier_camera` (a track broken in two still counts once,
    start to end). A pass without stored tracks (written before
    pass_tracks existed) is skipped.
    """
    out = []
    for p in passes:
        if p.get("status", "final") != "final":
            continue
        ts = tracks.get(p["pass_uid"])
        if not ts:
            continue
        first = min(t["start"] for t in ts)
        end = max(t["end"] for t in ts)
        head = [t for t in ts if t["camera"] == barrier_camera]
        barrier = (max(t["end"] for t in head) - min(t["start"] for t in head)) if head else None
        codes = p.get("codes")
        if codes is None:
            codes = tuple((p.get("container_codes") or "").split())
        out.append(PassTiming(
            p["pass_uid"], datetime.fromtimestamp(first), datetime.fromtimestamp(p["entry_time"]),
            datetime.fromtimestamp(end), barrier, p.get("vehicle_type") or "unknown",
            p.get("capacity_class") or "unknown", len(codes)))
    return sorted(out, key=lambda t: t.entry)


def hour_start(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def is_day(t: datetime) -> bool:
    return t.hour in DAY_HOURS


@dataclass(frozen=True)
class HourRow:
    hour: datetime                  # start of the clock hour
    recorded_s: float               # seconds of it actually recorded
    n: int                          # trucks with their checkpoint time in it
    by_class: dict[str, int]

    @property
    def per_minute(self) -> Optional[float]:
        return self.n / (self.recorded_s / 60) if self.recorded_s > 0 else None


def hourly_counts(timings: Sequence[PassTiming], coverage: Mapping[datetime, float]) -> list[HourRow]:
    """Trucks per recorded clock hour (by checkpoint time), with the capacity-class mix.

    `coverage`: {hour start: seconds recorded in that hour}. Hours not in
    it were not recorded and are left out; a truck in such an hour is
    ignored (it cannot happen with replayed recordings, but keeps the
    rates honest if the inputs disagree).
    """
    by_hour: dict[datetime, list[PassTiming]] = defaultdict(list)
    for t in timings:
        by_hour[hour_start(t.entry)].append(t)
    return [HourRow(h, s, len(by_hour[h]), dict(Counter(t.capacity_class for t in by_hour[h])))
            for h, s in sorted(coverage.items())]


def rate_per_minute(rows: Sequence[HourRow]) -> Optional[float]:
    """Trucks per minute over the recorded time of `rows`."""
    secs = sum(r.recorded_s for r in rows)
    return sum(r.n for r in rows) / (secs / 60) if secs > 0 else None


def interarrivals(timings: Sequence[PassTiming], coverage: Mapping[datetime, float]) -> list[float]:
    """Seconds between consecutive checkpoint times, only within fully recorded stretches.

    A gap that spans an hour not fully recorded (PC off, part hour) is
    dropped: the trucks that came while nothing recorded would make it look
    long.
    """
    full = {h for h, s in coverage.items() if s >= 3600 - 60}
    out = []
    for a, b in zip(timings, timings[1:]):
        h, last = hour_start(a.entry), hour_start(b.entry)
        ok = True
        while h <= last:
            if h not in full:
                ok = False
                break
            h += timedelta(hours=1)
        if ok:
            out.append((b.entry - a.entry).total_seconds())
    return out


def cv(xs: Sequence[float]) -> Optional[float]:
    """Coefficient of variation (sample std / mean); 1 for exponential gaps (Poisson)."""
    if len(xs) < 2 or statistics.fmean(xs) == 0:
        return None
    return statistics.stdev(xs) / statistics.fmean(xs)


def dispersion_index(timings: Sequence[PassTiming], coverage: Mapping[datetime, float],
                     bin_s: float = 600.0) -> Optional[float]:
    """Variance / mean of truck counts per `bin_s` bin over fully recorded hours.

    About 1 for a Poisson process at a steady rate; above 1 when trucks
    come in bunches or the rate changes over the hours (which it does
    between day and night — compare within day hours only for the Poisson
    question).
    """
    full = sorted(h for h, s in coverage.items() if s >= 3600 - 60)
    if not full:
        return None
    per_hour = int(3600 // bin_s)
    counts = {(h, i): 0 for h in full for i in range(per_hour)}
    for t in timings:
        h = hour_start(t.entry)
        key = (h, int((t.entry - h).total_seconds() // bin_s))
        if key in counts:
            counts[key] += 1
    xs = list(counts.values())
    m = statistics.fmean(xs)
    if len(xs) < 2 or m == 0:
        return None
    return statistics.variance(xs) / m


@dataclass(frozen=True)
class Summary:
    n: int
    median: Optional[float]
    mean: Optional[float]
    p90: Optional[float]


def summarise(xs: Sequence[float]) -> Summary:
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return Summary(0, None, None, None)
    p90 = xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)]
    return Summary(len(xs), statistics.median(xs), statistics.fmean(xs), p90)


def by_group(timings: Sequence[PassTiming], value, key=lambda t: t.capacity_class
             ) -> dict[str, Summary]:
    """{group: Summary of value(t)} — e.g. barrier time per capacity class."""
    groups: dict[str, list[float]] = defaultdict(list)
    for t in timings:
        v = value(t)
        if v is not None:
            groups[key(t)].append(v)
    return {g: summarise(vs) for g, vs in sorted(groups.items())}
