"""Robustness checks for Unstrict FIFO v2 under different gate assumptions.

The headline evaluation (scripts/run_evaluation.py) uses strongly
heterogeneous gates. This script re-runs StrictFIFO, the FYP1 router and
v2 under three scenarios to show how far the result depends on that
assumption. All scenarios have the same total capacity (3 trucks/min),
so the same arrival rates apply.

    base    A=180 / B=90 / C=30 s, std = mean/6   (the evaluation config)
    mild    A=90  / B=60 / C=45 s, std 15/10/8    (the live-demo config)
    noisy   A=180 / B=90 / C=30 s, std = mean/2   (high service-time variance)

v2 keeps its default lookahead throughout: the point is how it holds up
when conditions differ from the ones it was tuned for. Note the simulator
truncates service times at 0.1 s; in `noisy` about 2% of draws hit that
floor, raising the true means by roughly 1%.

Outputs:
    results/robustness.csv     — one row per (scenario, router, rate, seed)
    results/robustness.md      — summary tables for the report
"""

import csv
import dataclasses
import sys
import time
from dataclasses import asdict
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation import pairwise_tests, run_evaluation, summarize_cells
from src.router import StrictFIFORouter, UnstrictFIFORouter, UnstrictFIFOv2Router
from src.simulation import GateConfig

from run_evaluation import CONFIG


SCENARIOS = {
    "base": (GateConfig("A", 180, 30), GateConfig("B", 90, 15), GateConfig("C", 30, 5)),
    "mild": (GateConfig("A", 90, 15), GateConfig("B", 60, 10), GateConfig("C", 45, 8)),
    "noisy": (GateConfig("A", 180, 90), GateConfig("B", 90, 45), GateConfig("C", 30, 15)),
}
METRICS = ("mean_sojourn", "p95_sojourn", "mean_wait")


# Module-level builders (picklable for multiprocessing).
def build_strict_fifo(seed: int):
    return StrictFIFORouter()

def build_unstrict_fifo(seed: int):
    return UnstrictFIFORouter(alpha=0.7)

def build_unstrict_fifo_v2(seed: int):
    return UnstrictFIFOv2Router()


ROUTERS = {
    "StrictFIFO": build_strict_fifo,
    "UnstrictFIFO": build_unstrict_fifo,
    "UnstrictFIFOv2": build_unstrict_fifo_v2,
}
VERDICT = {"a<b": "v2 better", "a>b": "v2 worse", "ns": "ns"}


def scenario_table(name: str, records) -> list[str]:
    cells = {(c.router_name, c.arrival_rate): c for c in summarize_cells(records)}
    tests = {
        (t.router_b, t.arrival_rate, t.metric): VERDICT[t.direction]
        for m in METRICS
        for t in pairwise_tests(records, baseline_router="UnstrictFIFOv2", metric=m)
    }
    lines = [
        f"## Scenario: {name}",
        "",
        "| Rate | Metric | StrictFIFO | UnstrictFIFO | v2 | v2 vs Strict | v2 vs FYP1 | Significance (vs Strict / vs FYP1) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rate in CONFIG.arrival_rates:
        for metric in METRICS:
            s, a, b = (getattr(cells[(r, rate)], metric) for r in ROUTERS)
            lines.append(
                f"| {rate} | {metric} | {s:.1f} | {a:.1f} | {b:.1f} | "
                f"{1 - b / s:.1%} | {1 - b / a:.1%} | "
                f"{tests[('StrictFIFO', rate, metric)]} / {tests[('UnstrictFIFO', rate, metric)]} |"
            )
    held = [r.n_held / max(r.n_arrivals, 1) for r in records if r.router_name == "UnstrictFIFOv2"]
    lines += ["", f"v2 held {sum(held) / len(held):.1%} of trucks on average in this scenario.", ""]
    return lines


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Unstrict FIFO v2 robustness checks",
        "",
        f"{CONFIG.n_seeds} seeds per cell; Bonferroni-corrected Mann-Whitney tests vs v2.",
        "Positive percentages mean v2 is lower (better).",
        "",
    ]
    rows = []
    t0 = time.time()
    for name, gates in SCENARIOS.items():
        print(f"Scenario {name}...")
        records = run_evaluation(ROUTERS, dataclasses.replace(CONFIG, gates=gates), progress=False)
        rows += [{"scenario": name, **asdict(r)} for r in records]
        report += scenario_table(name, records)
    print(f"Done in {time.time() - t0:.1f}s")

    with (out_dir / "robustness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "robustness.md").write_text("\n".join(report))
    print(f"Wrote {out_dir / 'robustness.csv'} and {out_dir / 'robustness.md'}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
