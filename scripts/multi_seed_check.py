"""Multi-seed sanity check.

Runs StrictFIFO vs UnstrictFIFO across N seeds at each arrival rate.
Reports mean ± SE so we can tell signal from noise before committing to
the full evaluation harness in Component 7.
"""

import random
import statistics
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.db import init_schema, make_engine
from src.ocr import MockOCRAdapter
from src.router import StrictFIFORouter, UnstrictFIFORouter
from src.simulation import DepotSimulation, GateConfig, SimulationConfig
from src.synthetic_images import CLEAN, ContainerImageGenerator


N_SEEDS = 10
ARRIVAL_RATES = [1.5, 1.8, 2.0, 2.2, 2.4] 
DURATION = 7200   # 1 simulated hour per run


def run_once(router_factory, arrival_rate, seed):
    """One simulation run; returns (mean_wait, p95_wait, n_completed)."""
    engine = make_engine("sqlite:///:memory:")
    init_schema(engine)
    sim = DepotSimulation(
        config=SimulationConfig(
            duration_seconds=DURATION,
            arrival_rate_per_minute=arrival_rate,
            seed=seed,
            image_config=CLEAN,
            gates=(
                GateConfig("A", mean_service_seconds=180.0, std_service_seconds=30.0),
                GateConfig("B", mean_service_seconds=90.0,  std_service_seconds=15.0),
                GateConfig("C", mean_service_seconds=30.0,  std_service_seconds=5.0),
            ),
        ),
        router=router_factory(),
        ocr=MockOCRAdapter(error_rate=0.05, rng=random.Random(seed)),
        engine=engine,
        image_generator=ContainerImageGenerator(rng=random.Random(seed)),
        fast_mode=True,
    )
    result = sim.run()
    waits = sorted(result.wait_times)
    if not waits:
        return float("nan"), float("nan"), result.n_completed
    mean_w = sum(waits) / len(waits)
    p95 = waits[int(0.95 * (len(waits) - 1))]
    return mean_w, p95, result.n_completed


def summarize(values):
    """Return (mean, standard_error_of_the_mean)."""
    if len(values) < 2:
        return values[0] if values else float("nan"), 0.0
    mean = statistics.mean(values)
    se = statistics.stdev(values) / (len(values) ** 0.5)
    return mean, se


ROUTERS = {
    "StrictFIFO":           lambda: StrictFIFORouter(),
    "UnstrictFIFO(α=0.7)":  lambda: UnstrictFIFORouter(alpha=0.7),
}


def main():
    print(f"{N_SEEDS} seeds per cell, {DURATION}s simulated per run\n")
    header = (
        f"{'rate':>5}  {'router':<22}  "
        f"{'mean_wait (±SE)':<22}  {'p95_wait (±SE)':<22}  {'mean_done':>10}"
    )
    print(header)
    print("-" * len(header))

    for rate in ARRIVAL_RATES:
        rate_block: dict[str, dict[str, tuple[float, float]]] = {}
        for router_name, router_factory in ROUTERS.items():
            means, p95s, dones = [], [], []
            for seed in range(N_SEEDS):
                m, p, d = run_once(router_factory, rate, seed)
                means.append(m)
                p95s.append(p)
                dones.append(d)
            mean_w, mean_w_se = summarize(means)
            p95_w, p95_w_se = summarize(p95s)
            mean_done = sum(dones) / len(dones)
            rate_block[router_name] = {
                "mean": (mean_w, mean_w_se),
                "p95": (p95_w, p95_w_se),
                "done": mean_done,
            }
            print(
                f"{rate:>5.1f}  {router_name:<22}  "
                f"{mean_w:>7.1f} ± {mean_w_se:<7.1f}     "
                f"{p95_w:>7.1f} ± {p95_w_se:<7.1f}     "
                f"{mean_done:>10.1f}"
            )

        # Interpret this rate.
        s = rate_block["StrictFIFO"]
        u = rate_block["UnstrictFIFO(α=0.7)"]
        diff_mean = s["mean"][0] - u["mean"][0]
        diff_se = (s["mean"][1] ** 2 + u["mean"][1] ** 2) ** 0.5
        # A difference is "real" if it exceeds 2 SE (rough 95% guideline).
        if diff_se == 0:
            verdict = "same"
        elif abs(diff_mean) < 2 * diff_se:
            verdict = "noise (within 2 SE)"
        elif diff_mean > 0:
            verdict = f"unstrict WINS by {diff_mean:.1f}s ({diff_mean/diff_se:.1f} SE)"
        else:
            verdict = f"strict wins by {-diff_mean:.1f}s ({-diff_mean/diff_se:.1f} SE)"
        print(f"        verdict: {verdict}\n")


if __name__ == "__main__":
    main()