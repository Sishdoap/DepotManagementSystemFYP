"""Tests for the SimPy simulation harness."""

import random
import pytest
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.db import drop_schema, init_schema, make_engine
from src.ocr import MockOCRAdapter
from src.router import (
    RandomRouter,
    RoundRobinRouter,
    ShortestQueueRouter,
    StrictFIFORouter,
    UnstrictFIFORouter,
)
from src.simulation import (
    DepotSimulation,
    GateConfig,
    SimulationConfig,
    SimulationResult,
)
from src.synthetic_images import CLEAN, ContainerImageGenerator


def make_sim(*, router, seed=0, duration=600, arrival_rate=2.0, engine=None):
    """Build a small simulation for testing."""
    if engine is None:
        engine = make_engine("sqlite:///:memory:")
        init_schema(engine)
    config = SimulationConfig(
        duration_seconds=duration,
        arrival_rate_per_minute=arrival_rate,
        seed=seed,
        image_config=CLEAN,
    )
    return DepotSimulation(
        config=config,
        router=router,
        ocr=MockOCRAdapter(error_rate=0.0, rng=random.Random(seed)),
        engine=engine,
        image_generator=ContainerImageGenerator(rng=random.Random(seed)),
        fast_mode=True,
    )


class TestBasicRun:
    def test_simulation_runs_to_completion(self):
        sim = make_sim(router=StrictFIFORouter())
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_arrivals_occur(self):
        sim = make_sim(router=StrictFIFORouter(), duration=600, arrival_rate=2.0)
        result = sim.run()
        # 10 minutes at 2/min = ~20 trucks expected.
        assert result.n_arrivals > 10
        assert result.n_arrivals < 40

    def test_completed_le_arrived(self):
        # At end of simulation, some trucks may still be queued or in service.
        sim = make_sim(router=StrictFIFORouter())
        result = sim.run()
        assert result.n_completed <= result.n_arrivals

    def test_wait_times_recorded(self):
        sim = make_sim(router=StrictFIFORouter())
        result = sim.run()
        assert all(w >= 0 for w in result.wait_times)

    def test_service_times_positive(self):
        sim = make_sim(router=StrictFIFORouter())
        result = sim.run()
        assert all(s > 0 for s in result.service_times)


class TestReproducibility:
    """Same seed -> identical results (NFR: reproducibility)."""

    def test_same_seed_identical_arrivals(self):
        r1 = make_sim(router=StrictFIFORouter(), seed=42).run()
        r2 = make_sim(router=StrictFIFORouter(), seed=42).run()
        assert r1.n_arrivals == r2.n_arrivals
        assert r1.n_completed == r2.n_completed

    def test_same_seed_identical_wait_times(self):
        r1 = make_sim(router=StrictFIFORouter(), seed=42).run()
        r2 = make_sim(router=StrictFIFORouter(), seed=42).run()
        assert r1.wait_times == r2.wait_times

    def test_different_seeds_differ(self):
        r1 = make_sim(router=StrictFIFORouter(), seed=1).run()
        r2 = make_sim(router=StrictFIFORouter(), seed=2).run()
        # At least one of arrivals or wait times should differ.
        assert r1.wait_times != r2.wait_times or r1.n_arrivals != r2.n_arrivals


class TestRouterIntegration:
    """All five routers should run without error."""

    @pytest.mark.parametrize("router_factory", [
        lambda: StrictFIFORouter(),
        lambda: UnstrictFIFORouter(alpha=0.7),
        lambda: RoundRobinRouter(),
        lambda: RandomRouter(rng=random.Random(0)),
        lambda: ShortestQueueRouter(),
    ])
    def test_router_completes_run(self, router_factory):
        sim = make_sim(router=router_factory())
        result = sim.run()
        assert result.n_arrivals > 0


class TestGateBehavior:
    def test_gates_have_positive_utilization(self):
        # Over a long-enough run, all gates should be used at least once.
        sim = make_sim(router=ShortestQueueRouter(), duration=1800, arrival_rate=3.0)
        result = sim.run()
        for gate_id, util in result.per_gate_utilization.items():
            assert util > 0, f"Gate {gate_id} was never used"

    def test_utilization_bounded(self):
        sim = make_sim(router=StrictFIFORouter())
        result = sim.run()
        for util in result.per_gate_utilization.values():
            assert 0 <= util <= 1


class TestUnstrictFIFOBehavior:
    """The interesting smoke test: does unstrict FIFO actually use Gate C more?

    Given our default gate configs (C is fastest at 45s mean vs A's 90s),
    unstrict FIFO should send Gate C noticeably more traffic than Gate A.
    """

    def test_unstrict_fifo_favors_faster_gate(self):
        sim = make_sim(
            router=UnstrictFIFORouter(alpha=0.7),
            duration=3600,
            arrival_rate=2.5,
            seed=0,
        )
        result = sim.run()
        # Gate C is configured to be fastest. Unstrict FIFO should send
        # noticeably more trucks there than to Gate A (the slowest).
        c_count = result.per_gate_throughput["C"]
        a_count = result.per_gate_throughput["A"]
        assert c_count > a_count, (
            f"Unstrict FIFO sent more trucks to slow gate A ({a_count}) than "
            f"to fast gate C ({c_count}) — algorithm or config is wrong"
        )


class TestDatabaseIntegration:
    def test_events_written(self):
        engine = make_engine("sqlite:///:memory:")
        init_schema(engine)
        sim = make_sim(router=StrictFIFORouter(), engine=engine)
        result = sim.run()

        # Query the database to confirm rows were written.
        from src.db import events
        from sqlalchemy import select
        with engine.begin() as conn:
            rows = conn.execute(select(events)).all()
        assert len(rows) == result.n_arrivals

    def test_container_readings_written(self):
        engine = make_engine("sqlite:///:memory:")
        init_schema(engine)
        sim = make_sim(router=StrictFIFORouter(), engine=engine)
        result = sim.run()

        from src.db import containers
        from sqlalchemy import select
        with engine.begin() as conn:
            rows = conn.execute(select(containers)).all()
        # Container readings happen at gate assignment, so they match completed
        # gate-service starts. Should equal the number of trucks that made it
        # to a gate, which is >= n_completed and <= n_arrivals.
        assert len(rows) >= result.n_completed