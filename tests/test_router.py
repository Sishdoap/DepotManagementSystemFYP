"""Tests for routing algorithms."""

import random
import pytest
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.router import (
    Decision,
    GateState,
    RandomRouter,
    RoundRobinRouter,
    Router,
    RouterState,
    ShortestQueueRouter,
    StrictFIFORouter,
    UnstrictFIFORouter,
)


def make_state(
    *gate_specs,
    global_queue: int = 1,
    head_wait: float = 0.0,
    now: float = 1000.0,
) -> RouterState:
    """Build a RouterState from compact tuples: (id, busy, queue_len, throughput)."""
    gates = [
        GateState(gate_id=gid, is_busy=busy, queue_length=ql, recent_throughput=tp)
        for gid, busy, ql, tp in gate_specs
    ]
    return RouterState(
        gates=gates,
        global_queue_length=global_queue,
        head_truck_wait_seconds=head_wait,
        now=now,
    )


# ============================================================
# Common contract tests — every router must satisfy these
# ============================================================

ALL_ROUTERS: list[tuple[str, callable]] = [
    ("unstrict", lambda: UnstrictFIFORouter()),
    ("strict", lambda: StrictFIFORouter()),
    ("round_robin", lambda: RoundRobinRouter()),
    ("random", lambda: RandomRouter(rng=random.Random(0))),
    ("shortest_queue", lambda: ShortestQueueRouter()),
]


@pytest.mark.parametrize("name,factory", ALL_ROUTERS)
class TestRouterContract:
    def test_empty_queue_returns_none(self, name, factory):
        router = factory()
        state = make_state(("A", False, 0, 0.0), global_queue=0)
        assert router.choose(state) is None

    def test_all_busy_returns_none(self, name, factory):
        router = factory()
        state = make_state(
            ("A", True, 0, 5.0),
            ("B", True, 0, 4.0),
            ("C", True, 0, 6.0),
        )
        assert router.choose(state) is None

    def test_decision_picks_free_gate(self, name, factory):
        router = factory()
        state = make_state(
            ("A", True, 0, 5.0),
            ("B", False, 1, 4.0),
            ("C", True, 0, 6.0),
        )
        decision = router.choose(state)
        # Skip round-robin: rotation may target a busy gate (intentional).
        if name == "round_robin":
            # If RR returns a decision, it must be the free gate.
            if decision is not None:
                assert decision.gate_id == "B"
        else:
            assert decision is not None
            assert decision.gate_id == "B"

    def test_decision_has_reason(self, name, factory):
        router = factory()
        state = make_state(("A", False, 0, 5.0))
        decision = router.choose(state)
        if decision is not None:
            assert isinstance(decision.reason, str)
            assert len(decision.reason) > 0


# ============================================================
# Unstrict FIFO — the research contribution
# ============================================================

class TestUnstrictFIFO:
    def test_starvation_guard_fires(self):
        router = UnstrictFIFORouter(max_wait_seconds=60)
        state = make_state(
            ("A", False, 0, 5.0),
            ("B", False, 5, 10.0),    # B is way faster but truck has waited
            ("C", False, 2, 7.0),
            head_wait=300.0,
        )
        decision = router.choose(state)
        assert decision is not None
        assert "starvation" in decision.reason.lower()

    def test_bootstrap_picks_unseen_gate(self):
        # Gate C has no throughput history yet — should be picked over A/B.
        router = UnstrictFIFORouter(alpha=0.7)
        state = make_state(
            ("A", False, 0, 5.0),
            ("B", False, 0, 4.0),
            ("C", False, 0, 0.0),
        )
        decision = router.choose(state)
        assert decision.gate_id == "C"
        assert "bootstrap" in decision.reason.lower()

    def test_pure_throughput_picks_fastest(self):
        # alpha=1.0 ignores queue length entirely.
        router = UnstrictFIFORouter(alpha=1.0)
        state = make_state(
            ("A", False, 0, 3.0),
            ("B", False, 5, 10.0),    # busiest queue but fastest
            ("C", False, 1, 6.0),
        )
        decision = router.choose(state)
        assert decision.gate_id == "B"

    def test_pure_queue_picks_shortest(self):
        # alpha=0.0 degenerates to shortest-queue behavior.
        router = UnstrictFIFORouter(alpha=0.0)
        state = make_state(
            ("A", False, 4, 10.0),    # fastest, but queue is long
            ("B", False, 0, 3.0),     # slowest, but no queue
            ("C", False, 2, 6.0),
        )
        decision = router.choose(state)
        assert decision.gate_id == "B"

    def test_balanced_tradeoff(self):
        # With alpha=0.5 and a clear winner on both dims, pick the obvious gate.
        router = UnstrictFIFORouter(alpha=0.5)
        state = make_state(
            ("A", False, 5, 2.0),     # slow AND congested
            ("B", False, 0, 8.0),     # fast AND empty — strict winner
            ("C", False, 2, 5.0),     # middling
        )
        decision = router.choose(state)
        assert decision.gate_id == "B"

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            UnstrictFIFORouter(alpha=1.5)
        with pytest.raises(ValueError):
            UnstrictFIFORouter(alpha=-0.1)

    def test_reason_includes_scoring_details(self):
        # When unstrict FIFO actually scores (not bootstrap/starvation),
        # the reason should expose the score and the throughput/queue values
        # for transparency — supports your "visualisation of routing decisions"
        # requirement.
        router = UnstrictFIFORouter(alpha=0.7)
        state = make_state(
            ("A", False, 3, 4.0),
            ("B", False, 1, 8.0),
            ("C", False, 2, 5.0),
        )
        decision = router.choose(state)
        assert "score" in decision.reason.lower()
        assert "tpm" in decision.reason.lower()


# ============================================================
# Baselines
# ============================================================

class TestStrictFIFO:
    def test_picks_first_free_alphabetically(self):
        router = StrictFIFORouter()
        state = make_state(
            ("C", False, 0, 5.0),
            ("A", False, 0, 5.0),
            ("B", False, 0, 5.0),
        )
        assert router.choose(state).gate_id == "A"

    def test_skips_busy_gates(self):
        router = StrictFIFORouter()
        state = make_state(
            ("A", True, 0, 5.0),
            ("B", False, 0, 5.0),
            ("C", False, 0, 5.0),
        )
        assert router.choose(state).gate_id == "B"


class TestRoundRobin:
    def test_cycles_through_gates(self):
        router = RoundRobinRouter()
        gates = [("A", False, 0, 5.0), ("B", False, 0, 5.0), ("C", False, 0, 5.0)]
        # Three calls should yield A, B, C in order.
        assert router.choose(make_state(*gates)).gate_id == "A"
        assert router.choose(make_state(*gates)).gate_id == "B"
        assert router.choose(make_state(*gates)).gate_id == "C"
        # Fourth call wraps to A.
        assert router.choose(make_state(*gates)).gate_id == "A"

    def test_waits_if_rotation_target_busy(self):
        router = RoundRobinRouter()
        # Cursor starts at 0 -> A. But A is busy.
        state = make_state(
            ("A", True, 0, 5.0),
            ("B", False, 0, 5.0),
            ("C", False, 0, 5.0),
        )
        # Faithful round-robin waits rather than skipping ahead.
        assert router.choose(state) is None


class TestRandom:
    def test_reproducible_with_seed(self):
        gates = [("A", False, 0, 5.0), ("B", False, 0, 5.0), ("C", False, 0, 5.0)]
        r1 = RandomRouter(rng=random.Random(42))
        r2 = RandomRouter(rng=random.Random(42))
        seq1 = [r1.choose(make_state(*gates)).gate_id for _ in range(20)]
        seq2 = [r2.choose(make_state(*gates)).gate_id for _ in range(20)]
        assert seq1 == seq2

    def test_distribution_roughly_uniform(self):
        # Over many trials, each gate should be chosen ~1/3 of the time.
        router = RandomRouter(rng=random.Random(0))
        gates = [("A", False, 0, 5.0), ("B", False, 0, 5.0), ("C", False, 0, 5.0)]
        counts = {"A": 0, "B": 0, "C": 0}
        N = 3000
        for _ in range(N):
            counts[router.choose(make_state(*gates)).gate_id] += 1
        # Each should be within ~5% of 1/3.
        for c in counts.values():
            assert abs(c / N - 1 / 3) < 0.05


class TestShortestQueue:
    def test_picks_shortest(self):
        router = ShortestQueueRouter()
        state = make_state(
            ("A", False, 5, 5.0),
            ("B", False, 2, 5.0),
            ("C", False, 7, 5.0),
        )
        assert router.choose(state).gate_id == "B"

    def test_ties_broken_alphabetically(self):
        router = ShortestQueueRouter()
        state = make_state(
            ("B", False, 1, 5.0),
            ("A", False, 1, 5.0),
            ("C", False, 1, 5.0),
        )
        assert router.choose(state).gate_id == "A"