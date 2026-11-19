"""Tests for arrival and checkpoint-time measurement (src/arrivals.py)."""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.arrivals import (PassTiming, by_group, cv, dispersion_index, hourly_counts, interarrivals,
                          is_day, pass_timings, rate_per_minute, summarise)

T0 = datetime(2026, 11, 14, 10, 0, 0)


def timing(offset_s, cls="medium", barrier=30.0, wait=10.0, uid=None):
    entry = T0 + timedelta(seconds=offset_s)
    return PassTiming(uid or f"p{offset_s}", entry - timedelta(seconds=wait), entry,
                      entry + timedelta(seconds=60), barrier, "container_carrier", cls, 1)


def test_pass_timings_from_db_rows():
    ts = T0.timestamp()
    passes = [
        {"pass_uid": "a", "status": "final", "entry_time": ts + 5, "vehicle_type": "box_truck",
         "capacity_class": "medium", "codes": ()},
        {"pass_uid": "b", "status": "provisional", "entry_time": ts, "vehicle_type": None,
         "capacity_class": None, "codes": ()},
        {"pass_uid": "c", "status": "final", "entry_time": ts + 100, "vehicle_type": None,
         "capacity_class": None, "container_codes": "EMCU6461774 MSKU3398188"},
        {"pass_uid": "old", "status": "final", "entry_time": ts, "codes": ()},   # no tracks stored
    ]
    tracks = {
        "a": [{"camera": "557", "start": ts - 40, "end": ts + 20},
              {"camera": "561", "start": ts + 5, "end": ts + 25},
              {"camera": "561", "start": ts + 30, "end": ts + 45}],    # broken track: one span
        "b": [{"camera": "561", "start": ts, "end": ts + 9}],
        "c": [{"camera": "558", "start": ts + 100, "end": ts + 130}],
    }
    got = pass_timings(passes, tracks)
    assert [t.pass_uid for t in got] == ["a", "c"]
    a, c = got
    assert a.barrier_s == 40 and a.wait_s == 45 and a.through_s == 85
    assert a.first_seen == datetime.fromtimestamp(ts - 40)
    assert c.barrier_s is None and c.n_codes == 2
    assert c.vehicle_type == "unknown" and c.capacity_class == "unknown"
    assert c.wait_s == 0


def test_hourly_counts_divide_by_recorded_time():
    ts = [timing(60), timing(120, "heavy"), timing(3600 + 30), timing(7200 + 10)]
    cov = {T0: 3600.0, T0 + timedelta(hours=1): 1800.0}     # 12:00 not recorded
    rows = hourly_counts(ts, cov)
    assert [(r.hour.hour, r.n) for r in rows] == [(10, 2), (11, 1)]
    assert rows[0].by_class == {"medium": 1, "heavy": 1}
    assert rows[0].per_minute == pytest.approx(2 / 60)
    assert rows[1].per_minute == pytest.approx(1 / 30)
    assert rate_per_minute(rows) == pytest.approx(3 / 90)
    assert rate_per_minute([]) is None


def test_interarrivals_skip_gaps_over_unrecorded_hours():
    ts = [timing(0), timing(100), timing(3600 + 50), timing(3 * 3600 + 5), timing(3 * 3600 + 65)]
    cov = {T0: 3600.0, T0 + timedelta(hours=1): 3600.0,
           T0 + timedelta(hours=2): 1200.0,                   # part hour
           T0 + timedelta(hours=3): 3600.0}
    assert interarrivals(ts, cov) == [100, 3550, 60]


def test_poisson_arrivals_have_cv_and_dispersion_near_one():
    rng = random.Random(1)
    t, ts = 0.0, []
    while t < 100 * 3600:                                     # 600 ten-minute bins
        t += rng.expovariate(1 / 120)                         # 0.5 per minute
        ts.append(timing(t))
    cov = {T0 + timedelta(hours=h): 3600.0 for h in range(100)}
    ts = [x for x in ts if x.entry < T0 + timedelta(hours=100)]
    assert cv(interarrivals(ts, cov)) == pytest.approx(1.0, abs=0.1)
    assert dispersion_index(ts, cov) == pytest.approx(1.0, abs=0.2)


def test_regular_arrivals_have_low_cv_and_dispersion():
    ts = [timing(i * 120) for i in range(300)]
    cov = {T0 + timedelta(hours=h): 3600.0 for h in range(10)}
    assert cv(interarrivals(ts, cov)) == pytest.approx(0.0)
    assert dispersion_index(ts, cov) == pytest.approx(0.0)
    assert cv([5.0]) is None and dispersion_index(ts, {}) is None


def test_summaries_by_class():
    ts = [timing(i, "heavy", barrier=b) for i, b in enumerate((10, 20, 30, 40, 100))]
    ts += [timing(10, "light", barrier=None)]
    got = by_group(ts, lambda t: t.barrier_s)
    assert set(got) == {"heavy"}
    h = got["heavy"]
    assert (h.n, h.median, h.mean, h.p90) == (5, 30, 40, 100)
    assert summarise([]).n == 0 and summarise([]).median is None


def test_day_is_seven_to_nineteen():
    assert is_day(T0.replace(hour=7)) and is_day(T0.replace(hour=18, minute=59))
    assert not is_day(T0.replace(hour=19)) and not is_day(T0.replace(hour=6, minute=59))
