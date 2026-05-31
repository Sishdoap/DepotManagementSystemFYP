"""Multi-seed evaluation harness for routing algorithms.

Runs many simulations across a grid of (router, arrival_rate, seed) and
produces statistically defensible comparisons:

  - bootstrap 95% confidence intervals on mean and p95 wait times
  - Mann-Whitney U tests pairwise vs unstrict FIFO (Bonferroni-corrected)
  - per-run raw data persisted to CSV for downstream plotting

Designed for headless batch use; the dashboard (Component 8) does live
visualization, this does post-hoc statistical analysis.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.image_source import SyntheticImageSource
from src.db import init_schema, make_engine
from src.ocr import MockOCRAdapter
from src.router import Router
from src.simulation import (
    DepotSimulation,
    GateConfig,
    SimulationConfig,
)
from src.synthetic_images import CLEAN, ContainerImageGenerator


# --- Configuration ---

@dataclass(frozen=True)
class EvaluationConfig:
    """Knobs for the evaluation grid."""
    arrival_rates: tuple[float, ...]
    n_seeds: int
    duration_seconds: float = 7200.0
    gates: tuple[GateConfig, ...] = (
        GateConfig("A", mean_service_seconds=180.0, std_service_seconds=30.0),
        GateConfig("B", mean_service_seconds=90.0, std_service_seconds=15.0),
        GateConfig("C", mean_service_seconds=30.0, std_service_seconds=5.0),
    )
    ocr_error_rate: float = 0.05
    n_workers: int = 0     # 0 = auto (os.cpu_count())


# --- Per-run output ---

@dataclass
class RunRecord:
    """One row of results per (router, rate, seed)."""
    router_name: str
    arrival_rate: float
    seed: int
    n_arrivals: int
    n_completed: int
    mean_wait: float
    p95_wait: float
    p99_wait: float
    max_wait: float
    util_A: float
    util_B: float
    util_C: float
    throughput_A: int
    throughput_B: int
    throughput_C: int
    n_ocr_valid: int
    n_recoveries: int


# --- The worker function (must be module-level for multiprocessing) ---

def _run_one(args) -> RunRecord:
    """Run a single simulation. Used by the multiprocessing pool.

    Args is a tuple because pool.imap_unordered passes one positional arg.
    """
    router_name, router_builder, rate, seed, eval_config = args

    engine = make_engine("sqlite:///:memory:")
    init_schema(engine)

    sim = DepotSimulation(
        config=SimulationConfig(
            duration_seconds=eval_config.duration_seconds,
            arrival_rate_per_minute=rate,
            seed=seed,
            image_config=CLEAN,
            gates=eval_config.gates,
        ),
        router=router_builder(seed),
        ocr=MockOCRAdapter(error_rate=eval_config.ocr_error_rate, rng=random.Random(seed)),
        engine=engine,
        image_source=SyntheticImageSource(ContainerImageGenerator(rng=random.Random(seed))),
        fast_mode=True,
    )
    r = sim.run()

    waits = sorted(r.wait_times) if r.wait_times else [0.0]
    n = len(waits)

    def pct(p):
        if n == 0:
            return 0.0
        idx = min(int(p * (n - 1)), n - 1)
        return waits[idx]

    return RunRecord(
        router_name=router_name,
        arrival_rate=rate,
        seed=seed,
        n_arrivals=r.n_arrivals,
        n_completed=r.n_completed,
        mean_wait=statistics.mean(waits),
        p95_wait=pct(0.95),
        p99_wait=pct(0.99),
        max_wait=max(waits),
        util_A=r.per_gate_utilization.get("A", 0.0),
        util_B=r.per_gate_utilization.get("B", 0.0),
        util_C=r.per_gate_utilization.get("C", 0.0),
        throughput_A=r.per_gate_throughput.get("A", 0),
        throughput_B=r.per_gate_throughput.get("B", 0),
        throughput_C=r.per_gate_throughput.get("C", 0),
        n_ocr_valid=r.n_ocr_valid,
        n_recoveries=r.n_recoveries,
    )


# --- The main evaluation runner ---

# Type alias: a router builder takes a seed and returns a configured router.
# Seeded routers (like RandomRouter) use the seed; others ignore it.
RouterBuilder = Callable[[int], Router]


def run_evaluation(
    routers: dict[str, RouterBuilder],
    eval_config: EvaluationConfig,
    progress: bool = True,
) -> list[RunRecord]:
    """Run the full grid in parallel, return all per-run records.

    Args:
        routers: dict of router_name -> builder function (takes seed).
        eval_config: grid configuration.
        progress: print progress to stdout.

    Returns:
        List of RunRecord, one per (router, rate, seed) combination.
    """
    tasks = []
    for router_name, builder in routers.items():
        for rate in eval_config.arrival_rates:
            for seed in range(eval_config.n_seeds):
                tasks.append((router_name, builder, rate, seed, eval_config))

    n_workers = eval_config.n_workers or mp.cpu_count()
    n_workers = min(n_workers, len(tasks))

    if progress:
        print(f"Running {len(tasks)} simulations across {n_workers} workers...")

    results: list[RunRecord] = []
    if n_workers == 1:
        # Serial path is easier to debug and required if the worker
        # touches non-pickleable state.
        for i, task in enumerate(tasks):
            results.append(_run_one(task))
            if progress and (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(tasks)} complete")
    else:
        with mp.Pool(n_workers) as pool:
            for i, rec in enumerate(pool.imap_unordered(_run_one, tasks)):
                results.append(rec)
                if progress and (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(tasks)} complete")

    return results


# --- Statistics ---

@dataclass
class CellSummary:
    """Aggregate statistics for one (router, rate) cell across seeds."""
    router_name: str
    arrival_rate: float
    n_seeds: int
    mean_wait: float
    mean_wait_ci_lo: float
    mean_wait_ci_hi: float
    p95_wait: float
    p95_wait_ci_lo: float
    p95_wait_ci_hi: float


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Bootstrap CI for the mean. Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        m = float(arr[0]) if len(arr) else 0.0
        return m, m, m
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.asarray(boot_means)
    alpha = 1 - ci
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def summarize_cells(records: list[RunRecord]) -> list[CellSummary]:
    """One CellSummary per (router_name, arrival_rate)."""
    grouped: dict[tuple[str, float], list[RunRecord]] = {}
    for r in records:
        grouped.setdefault((r.router_name, r.arrival_rate), []).append(r)

    out = []
    for (name, rate), recs in sorted(grouped.items()):
        means = [r.mean_wait for r in recs]
        p95s = [r.p95_wait for r in recs]
        mean_m, mean_lo, mean_hi = bootstrap_ci(means)
        p95_m, p95_lo, p95_hi = bootstrap_ci(p95s)
        out.append(CellSummary(
            router_name=name,
            arrival_rate=rate,
            n_seeds=len(recs),
            mean_wait=mean_m,
            mean_wait_ci_lo=mean_lo,
            mean_wait_ci_hi=mean_hi,
            p95_wait=p95_m,
            p95_wait_ci_lo=p95_lo,
            p95_wait_ci_hi=p95_hi,
        ))
    return out


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (U statistic, two-sided p-value) using scipy."""
    from scipy.stats import mannwhitneyu
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    res = mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


@dataclass
class PairwiseTest:
    """One pairwise test result."""
    router_a: str
    router_b: str
    arrival_rate: float
    metric: str             # "mean_wait" or "p95_wait"
    n_seeds: int
    u_statistic: float
    p_value_raw: float
    p_value_corrected: float
    direction: str          # "a<b" if a is faster, "a>b" if b is faster, "ns" otherwise


def pairwise_tests(
    records: list[RunRecord],
    *,
    baseline_router: str,
    metric: str = "mean_wait",
    alpha: float = 0.05,
) -> list[PairwiseTest]:
    """Compare every non-baseline router against the baseline at each rate.

    Returns Bonferroni-corrected pairwise test results.
    """
    # Group records by (router, rate).
    grouped: dict[tuple[str, float], list[float]] = {}
    for r in records:
        val = getattr(r, metric)
        grouped.setdefault((r.router_name, r.arrival_rate), []).append(val)

    routers = sorted({r.router_name for r in records})
    rates = sorted({r.arrival_rate for r in records})
    if baseline_router not in routers:
        raise ValueError(f"baseline router '{baseline_router}' not in records")

    other_routers = [x for x in routers if x != baseline_router]
    n_comparisons = len(other_routers) * len(rates)

    out: list[PairwiseTest] = []
    for rate in rates:
        baseline_vals = grouped.get((baseline_router, rate), [])
        for other in other_routers:
            other_vals = grouped.get((other, rate), [])
            u, p = mann_whitney_u(baseline_vals, other_vals)
            p_corr = min(1.0, p * n_comparisons) if not np.isnan(p) else float("nan")
            if np.isnan(p_corr) or p_corr >= alpha:
                direction = "ns"
            else:
                m_a = statistics.mean(baseline_vals)
                m_b = statistics.mean(other_vals)
                direction = "a<b" if m_a < m_b else "a>b"
            out.append(PairwiseTest(
                router_a=baseline_router,
                router_b=other,
                arrival_rate=rate,
                metric=metric,
                n_seeds=min(len(baseline_vals), len(other_vals)),
                u_statistic=u,
                p_value_raw=p,
                p_value_corrected=p_corr,
                direction=direction,
            ))
    return out


# --- Persistence ---

def save_records_csv(records: list[RunRecord], path: str | Path) -> None:
    """Write per-run records to CSV."""
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def save_statistics_json(
    cells: list[CellSummary],
    tests: list[PairwiseTest],
    path: str | Path,
) -> None:
    """Write summary statistics + pairwise tests to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": [asdict(c) for c in cells],
        "pairwise_tests": [asdict(t) for t in tests],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)