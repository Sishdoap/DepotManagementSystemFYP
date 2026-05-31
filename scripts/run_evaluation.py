"""Run the full evaluation: 5 routers × 5 rates × 30 seeds."""

import random
import sys
import time
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation import (
    EvaluationConfig,
    pairwise_tests,
    run_evaluation,
    save_records_csv,
    save_statistics_json,
    summarize_cells,
)
from src.router import (
    RandomRouter,
    RoundRobinRouter,
    ShortestQueueRouter,
    StrictFIFORouter,
    UnstrictFIFORouter,
)


# Module-level builder functions — picklable, unlike lambdas.
def build_strict_fifo(seed: int):
    return StrictFIFORouter()

def build_unstrict_fifo(seed: int):
    return UnstrictFIFORouter(alpha=0.7)

def build_round_robin(seed: int):
    return RoundRobinRouter()

def build_random(seed: int):
    return RandomRouter(rng=random.Random(seed))

def build_shortest_queue(seed: int):
    return ShortestQueueRouter()


ROUTERS = {
    "StrictFIFO":    build_strict_fifo,
    "UnstrictFIFO":  build_unstrict_fifo,
    "RoundRobin":    build_round_robin,
    "Random":        build_random,
    "ShortestQueue": build_shortest_queue,
}


CONFIG = EvaluationConfig(
    arrival_rates=(1.5, 1.8, 2.0, 2.2, 2.4),
    n_seeds=30,
    duration_seconds=7200.0,
    ocr_error_rate=0.05,
)


def main():
    out_dir = Path("results")

    t0 = time.time()
    records = run_evaluation(ROUTERS, CONFIG, progress=True)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(records)} runs)")

    save_records_csv(records, out_dir / "results.csv")
    print(f"Wrote {out_dir / 'results.csv'}")

    cells = summarize_cells(records)
    tests_mean = pairwise_tests(records, baseline_router="UnstrictFIFO", metric="mean_wait")
    tests_p95 = pairwise_tests(records, baseline_router="UnstrictFIFO", metric="p95_wait")
    save_statistics_json(cells, tests_mean + tests_p95, out_dir / "statistics.json")
    print(f"Wrote {out_dir / 'statistics.json'}")

    # Quick console summary.
    print("\n--- Mean wait time, 95% CI (bootstrap) ---")
    routers_order = list(ROUTERS.keys())
    rates = sorted({c.arrival_rate for c in cells})
    header = f"{'rate':>5}  " + "  ".join(f"{r:>22}" for r in routers_order)
    print(header)
    for rate in rates:
        row = f"{rate:>5.1f}  "
        for rname in routers_order:
            c = next((c for c in cells if c.router_name == rname and c.arrival_rate == rate), None)
            if c is None:
                row += f"{'-':>22}  "
            else:
                row += f"{c.mean_wait:>7.1f}[{c.mean_wait_ci_lo:>5.1f},{c.mean_wait_ci_hi:>5.1f}]  "
        print(row)

    print("\n--- Pairwise vs UnstrictFIFO (mean_wait, Bonferroni-corrected) ---")
    for t in tests_mean:
        sig = "***" if t.direction != "ns" else "   "
        verdict = {
            "a<b": "Unstrict faster",
            "a>b": f"{t.router_b} faster",
            "ns":  "no significant difference",
        }[t.direction]
        print(
            f"  {sig} rate={t.arrival_rate:.1f}  vs {t.router_b:<14} "
            f"p_corr={t.p_value_corrected:.4f}  {verdict}"
        )


if __name__ == "__main__":
    main()