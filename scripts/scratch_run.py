# scratch_run.py
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

def run(router, seed=0):
    engine = make_engine("sqlite:///:memory:")
    init_schema(engine)
    config = SimulationConfig(
        duration_seconds=3600,
        arrival_rate_per_minute=2.5,
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

for router in [StrictFIFORouter(), UnstrictFIFORouter(alpha=0.7)]:
    r = run(router, seed=0)
    mean_wait = sum(r.wait_times) / len(r.wait_times) if r.wait_times else 0
    print(f"\n{r.router_name}")
    print(f"  arrivals: {r.n_arrivals}, completed: {r.n_completed}")
    print(f"  mean wait: {mean_wait:.1f}s")
    print(f"  per-gate throughput: {r.per_gate_throughput}")
    util_str = ", ".join(f"{g}: {u:.1%}" for g, u in r.per_gate_utilization.items())
    print(f"  per-gate utilization: {util_str}")