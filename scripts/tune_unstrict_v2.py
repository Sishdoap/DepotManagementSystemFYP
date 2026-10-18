"""Sensitivity sweep for UnstrictFIFOv2Router's `lookahead_seconds`.

Runs the evaluation grid (same gates, rates and warm-up as
scripts/run_evaluation.py) for several lookahead values plus the
StrictFIFO and FYP1 UnstrictFIFO references, and reports mean time in
system per cell. Use it to choose the default lookahead and as the
sensitivity-analysis table for the report.

Tuning uses a separate block of seeds (TUNING_SEED_OFFSET) from the
evaluation's, so the chosen default isn't fitted to the exact runs it is
then judged on.

Outputs:
    results/tuning_v2.csv   — one row per (router, rate, seed)
"""

import dataclasses
import sys
import time
from functools import partial
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation import run_evaluation, save_records_csv, summarize_cells
from src.router import StrictFIFORouter, UnstrictFIFORouter, UnstrictFIFOv2Router

from run_evaluation import CONFIG


LOOKAHEADS = (0.0, 30.0, 60.0, 90.0, 120.0, 180.0, 300.0)
TUNING_SEED_OFFSET = 1000
TUNING_CONFIG = dataclasses.replace(CONFIG, seed_offset=TUNING_SEED_OFFSET)


# Module-level builders (picklable for multiprocessing).
def build_strict_fifo(seed: int):
    return StrictFIFORouter()

def build_unstrict_fifo(seed: int):
    return UnstrictFIFORouter(alpha=0.7)

def build_v2(seed: int, *, lookahead: float):
    return UnstrictFIFOv2Router(lookahead_seconds=lookahead)


def main():
    routers = {"StrictFIFO": build_strict_fifo, "UnstrictFIFO": build_unstrict_fifo}
    for la in LOOKAHEADS:
        routers[f"v2@{la:g}s"] = partial(build_v2, lookahead=la)

    t0 = time.time()
    records = run_evaluation(routers, TUNING_CONFIG, progress=True)
    print(f"\nDone in {time.time() - t0:.1f}s ({len(records)} runs)")
    save_records_csv(records, Path("results") / "tuning_v2.csv")

    cells = {(c.router_name, c.arrival_rate): c for c in summarize_cells(records)}
    rates = sorted({r for _, r in cells})

    print("\n--- Mean time in system (s) [reduction vs StrictFIFO] ---")
    print(f"{'router':<14}" + "".join(f"{r:>18.1f}" for r in rates) + f"{'avg cut':>10}")
    for name in routers:
        row, cuts = f"{name:<14}", []
        for rate in rates:
            c = cells[(name, rate)]
            cut = 1 - c.mean_sojourn / cells[("StrictFIFO", rate)].mean_sojourn
            cuts.append(cut)
            row += f"{c.mean_sojourn:>9.1f} [{cut:>6.1%}]"
        print(row + f"{sum(cuts) / len(cuts):>10.1%}")


if __name__ == "__main__":
    main()
