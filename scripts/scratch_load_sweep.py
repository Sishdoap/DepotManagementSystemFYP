# scratch_load_sweep.py
import random
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
    
from src.db import init_schema, make_engine
from src.ocr import MockOCRAdapter
from src.router import UnstrictFIFORouter, StrictFIFORouter
from src.simulation import DepotSimulation, SimulationConfig
from src.synthetic_images import CLEAN, ContainerImageGenerator


def run(router, arrival_rate, seed=0):
    engine = make_engine("sqlite:///:memory:")
    init_schema(engine)
    config = SimulationConfig(
        duration_seconds=3600,
        arrival_rate_per_minute=arrival_rate,
        seed=seed,
        image_config=CLEAN,
    )
    sim = DepotSimulation(
        config=config,
        router=router,
        ocr=MockOCRAdapter(error_rate=0.05, rng=random.Random(seed)),
        engine=engine,
        image_generator=ContainerImageGenerator(rng=random.Random(seed)),
        fast_mode=True,
    )
    return sim.run()


print(f"{'rate':>6}  {'router':<22}  {'arr':>5}  {'done':>5}  {'mean_wait':>10}  {'p95_wait':>10}")
for rate in [2.0, 2.5, 3.0, 3.5, 4.0]:
    for router_fn in [lambda: StrictFIFORouter(), lambda: UnstrictFIFORouter(alpha=0.7)]:
        r = run(router_fn(), arrival_rate=rate)
        waits = sorted(r.wait_times)
        mean_w = sum(waits) / len(waits) if waits else 0
        p95 = waits[int(0.95 * len(waits))] if len(waits) >= 20 else float("nan")
        print(f"{rate:>6.1f}  {r.router_name:<22}  {r.n_arrivals:>5}  {r.n_completed:>5}  {mean_w:>10.1f}  {p95:>10.1f}")