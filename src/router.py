"""Gate routing algorithms for the depot traffic management system.

Defines a Router abstract base class and five concrete implementations:

    UnstrictFIFORouter      — main research contribution; throughput- and
                              queue-aware scoring with starvation guard.
    StrictFIFORouter        — baseline: head-of-queue to any free gate.
    RoundRobinRouter        — baseline: cycle gates A, B, C, A, B, C ...
    RandomRouter            — baseline: uniform random over gates.
    ShortestQueueRouter     — baseline: pick the gate with fewest trucks.

All routers are pure functions of state — they take a RouterState snapshot
and return a Decision. They do not own queues, gates, or the database;
the caller (simulator or dashboard) is responsible for state mutation.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# --- Data classes for state and decisions ---

@dataclass(frozen=True)
class GateState:
    """Snapshot of one gate at decision time."""
    gate_id: str
    is_busy: bool              # True if currently serving a truck
    queue_length: int          # trucks waiting at this gate (0 if global queue)
    recent_throughput: float   # trucks per minute over the configured window


@dataclass(frozen=True)
class RouterState:
    """Snapshot of the routing environment at decision time.

    The router consumes this to pick a gate. The caller (simulator or
    dashboard) is responsible for building it correctly.
    """
    gates: list[GateState]
    global_queue_length: int   # trucks waiting overall
    head_truck_wait_seconds: float    # how long the head-of-queue truck has waited
    now: float                 # current unix timestamp


@dataclass(frozen=True)
class Decision:
    """The router's choice for the next truck."""
    gate_id: str               # gate to assign the head truck to
    reason: str                # human-readable explanation (logged to events.routing_reason)


# --- Abstract base ---

class Router(ABC):
    """Every router must implement `choose`."""

    @abstractmethod
    def choose(self, state: RouterState) -> Optional[Decision]:
        """Pick a gate for the head-of-queue truck.

        Returns None if no gate is available (caller should wait).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logging and the evaluation chapter's plots."""


# --- Helpers ---

def _free_gates(state: RouterState) -> list[GateState]:
    return [g for g in state.gates if not g.is_busy]


def _min_max_normalize(values: list[float]) -> list[float]:
    """Scale values to [0, 1]. If all equal, return zeros."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


# --- Unstrict FIFO (the research contribution) ---

class UnstrictFIFORouter(Router):
    """Score-based gate selection with throughput awareness and starvation guard.

    Algorithm:
        1. If no gates are free, return None (caller waits).
        2. If the head truck has waited more than `max_wait_seconds`, send it
           to any free gate immediately (starvation guard).
        3. If any free gate has zero recent throughput history, send the
           truck there (bootstrap; gather data before scoring).
        4. Otherwise, compute for each free gate:
               score(g) = alpha * norm(throughput_g) - (1-alpha) * norm(queue_g)
           where norm() is min-max normalization across the free gates.
        5. Return the highest-scoring gate.

    The single knob `alpha` controls the throughput/queue tradeoff:
        alpha = 1.0  -> pure throughput maximization (send to fastest)
        alpha = 0.5  -> balanced
        alpha = 0.0  -> pure shortest-queue (degenerate; use that baseline)

    Recommended starting value: alpha = 0.7 (throughput-biased, with enough
    queue weight to prevent congestion on the fastest gate).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.7,
        max_wait_seconds: float = 300.0,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha
        self._max_wait = max_wait_seconds

    @property
    def name(self) -> str:
        return f"UnstrictFIFO(alpha={self._alpha})"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None

        free = _free_gates(state)
        if not free:
            return None

        # Starvation guard.
        if state.head_truck_wait_seconds > self._max_wait:
            chosen = free[0]
            return Decision(
                gate_id=chosen.gate_id,
                reason=f"starvation guard: head truck waited {state.head_truck_wait_seconds:.0f}s",
            )

        # Bootstrap: prefer any free gate with no throughput history.
        unseen = [g for g in free if g.recent_throughput == 0.0]
        if unseen:
            # Pick the one with the shortest queue among bootstrap candidates.
            chosen = min(unseen, key=lambda g: g.queue_length)
            return Decision(
                gate_id=chosen.gate_id,
                reason="bootstrap: gate has no recent throughput history",
            )

        # Score-based selection.
        throughputs = [g.recent_throughput for g in free]
        queues = [float(g.queue_length) for g in free]
        norm_thr = _min_max_normalize(throughputs)
        norm_q = _min_max_normalize(queues)

        scores: list[tuple[float, GateState, float, float]] = []
        for g, nt, nq in zip(free, norm_thr, norm_q):
            score = self._alpha * nt - (1.0 - self._alpha) * nq
            scores.append((score, g, nt, nq))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, chosen, nt, nq = scores[0]
        reason = (
            f"unstrict FIFO: score={best_score:.3f} "
            f"(thr_norm={nt:.2f} @ {chosen.recent_throughput:.2f}tpm, "
            f"q_norm={nq:.2f} @ {chosen.queue_length})"
        )
        return Decision(gate_id=chosen.gate_id, reason=reason)


# --- Baselines ---

class StrictFIFORouter(Router):
    """Head-of-queue assigned to whichever gate is free first.

    Among multiple free gates, picks deterministically by gate_id (alphabetical)
    so the baseline is reproducible.
    """

    @property
    def name(self) -> str:
        return "StrictFIFO"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None
        free = _free_gates(state)
        if not free:
            return None
        chosen = min(free, key=lambda g: g.gate_id)
        return Decision(gate_id=chosen.gate_id, reason="strict FIFO: first free gate")


class RoundRobinRouter(Router):
    """Cycle gates A, B, C, A, B, C, ... regardless of state.

    If the next-in-rotation gate is busy, the truck waits (this is faithful
    to round-robin's actual behavior; some implementations skip-on-busy
    which makes them effectively shortest-queue, defeating the comparison).
    """

    def __init__(self):
        self._cursor = 0

    @property
    def name(self) -> str:
        return "RoundRobin"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None
        if not state.gates:
            return None
        # Sort gates by gate_id for determinism.
        ordered = sorted(state.gates, key=lambda g: g.gate_id)
        target = ordered[self._cursor % len(ordered)]
        if target.is_busy:
            return None
        self._cursor += 1
        return Decision(
            gate_id=target.gate_id,
            reason=f"round-robin: rotation position {self._cursor - 1}",
        )


class RandomRouter(Router):
    """Uniform random over free gates.

    Takes an explicit random.Random for reproducibility (NFR).
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()

    @property
    def name(self) -> str:
        return "Random"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None
        free = _free_gates(state)
        if not free:
            return None
        chosen = self._rng.choice(free)
        return Decision(gate_id=chosen.gate_id, reason="random: uniform over free gates")


class ShortestQueueRouter(Router):
    """Send to the gate with the fewest waiting trucks. Ties broken by gate_id."""

    @property
    def name(self) -> str:
        return "ShortestQueue"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None
        free = _free_gates(state)
        if not free:
            return None
        chosen = min(free, key=lambda g: (g.queue_length, g.gate_id))
        return Decision(
            gate_id=chosen.gate_id,
            reason=f"shortest queue: {chosen.queue_length} waiting",
        )