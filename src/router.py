"""Gate routing algorithms for the depot traffic management system.

Defines a Router abstract base class and five concrete implementations:

    UnstrictFIFOv2Router    — FYP2 research contribution; plans the queue
                              against expected gate completion times and may
                              hold a truck for a faster gate.
    UnstrictFIFORouter      — FYP1 version; throughput- and queue-aware
                              scoring over free gates, with starvation guard.
    StrictFIFORouter        — baseline: head-of-queue to any free gate.
    RoundRobinRouter        — baseline: cycle gates A, B, C, A, B, C ...
    RandomRouter            — baseline: uniform random over gates.
    ShortestQueueRouter     — baseline: pick the gate with fewest trucks
                              (per-gate queues only; not evaluated).

All routers are pure functions of state — they take a RouterState snapshot
and return a Decision. They do not own queues, gates, or the database;
the caller (simulator or dashboard) is responsible for state mutation.
"""

from __future__ import annotations

import math
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
    mean_service_seconds: Optional[float] = None  # observed mean; None if never served
    busy_seconds: float = 0.0  # how long the current truck has been in service (0 if free)


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
    arrival_rate_per_minute: float = 0.0  # recent observed arrival rate


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
           to the free gate with the best recent throughput immediately
           (starvation guard).
        3. If any free gate has never served a truck, send the truck there
           (bootstrap; gather data before scoring). A gate that has served
           before but is quiet in the current window is *not* re-bootstrapped
           — it is scored, and its zero throughput ranks it last.
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
            chosen = max(free, key=lambda g: (g.recent_throughput, g.gate_id))
            return Decision(
                gate_id=chosen.gate_id,
                reason=f"starvation guard: head truck waited {state.head_truck_wait_seconds:.0f}s",
            )

        # Bootstrap: prefer any free gate that has never served a truck.
        unseen = [
            g for g in free
            if g.recent_throughput == 0.0 and g.mean_service_seconds is None
        ]
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


# --- Unstrict FIFO v2 (FYP2) ---

class UnstrictFIFOv2Router(Router):
    """Expected-completion routing that may hold a truck for a faster gate.

    Motivation. Every "always dispatch to some free gate" policy, including
    UnstrictFIFORouter, sends the head truck to slow Gate A whenever A is the
    only free gate — even if fast Gate C will be free in a few seconds and
    would clear the truck far sooner. This is the classic "slow server
    problem": with heterogeneous servers, the optimal policy leaves the slow
    server idle unless the queue is long enough to need it (a threshold
    policy; Lin & Kumar, 1984). This router approximates that threshold from
    observed data instead of a hand-set constant.

    Trucks still enter gates in strict arrival order (FIFO); what is
    "unstrict" is that the head truck is not obliged to take the first free
    gate.

    Algorithm, for the head truck:
        1. No trucks waiting or no free gate -> None.
        2. Bootstrap: a free gate that has never served a truck gets it, so
           every gate has a service-time estimate.
        3. Starvation guard: if the head truck has waited longer than
           `max_wait_seconds`, send it to the fastest free gate now.
        4. Plan. Each gate g becomes available at
               avail_g = now                                   (free)
               avail_g = now + max(min_remaining, S_g - busy_g)  (busy)
           where S_g is g's observed mean service time and busy_g how long
           its current truck has been in service. Then plan the queue
           greedily, in order: each truck takes the gate with the earliest
           expected completion avail_g + S_g, and that gate's availability
           advances by S_g. The plan covers the trucks waiting now plus
           `lookahead_seconds` worth of expected arrivals (from the recent
           arrival rate), so a free slow gate is used when the near-future
           load will need it, not only the trucks already queued.
        5. If the plan puts any truck on a currently free gate, dispatch the
           head truck to the fastest such gate. Otherwise return None — the
           truck waits for a busy, faster gate, and the dispatcher re-plans
           at the next arrival or departure.

    The single knob `lookahead_seconds` sets how eagerly slow gates are used:
        lookahead = 0     -> plan only for trucks already waiting (most
                             patient; holds the most)
        lookahead -> inf  -> plan for an ever-growing queue, so every free
                             gate is used (degenerates to fastest-free-gate)

    A busy gate that has never completed a truck has no estimate yet; it is
    treated as unavailable when planning.
    """

    def __init__(
        self,
        *,
        lookahead_seconds: float = 90.0,   # tuned on held-out seeds (scripts/tune_unstrict_v2.py)
        max_wait_seconds: float = 300.0,
        min_remaining_seconds: float = 1.0,
        max_plan_trucks: int = 200,
    ):
        if lookahead_seconds < 0:
            raise ValueError(f"lookahead_seconds must be >= 0, got {lookahead_seconds}")
        self._lookahead = lookahead_seconds
        self._max_wait = max_wait_seconds
        self._min_remaining = min_remaining_seconds
        self._max_plan_trucks = max_plan_trucks

    @property
    def name(self) -> str:
        return f"UnstrictFIFOv2(lookahead={self._lookahead:g}s)"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None

        free = _free_gates(state)
        if not free:
            return None

        # Bootstrap: learn every gate's speed before planning with it.
        unseen = [g for g in free if g.mean_service_seconds is None]
        if unseen:
            chosen = min(unseen, key=lambda g: g.gate_id)
            return Decision(
                gate_id=chosen.gate_id,
                reason="bootstrap: gate has no service-time history",
            )

        if state.head_truck_wait_seconds > self._max_wait:
            chosen = min(free, key=lambda g: (g.mean_service_seconds, g.gate_id))
            return Decision(
                gate_id=chosen.gate_id,
                reason=f"starvation guard: head truck waited {state.head_truck_wait_seconds:.0f}s",
            )

        # Expected availability of every gate we can plan with.
        service: dict[str, float] = {}
        avail: dict[str, float] = {}
        for g in state.gates:
            if g.mean_service_seconds is None:
                continue    # busy, never completed: unknown, leave out of the plan
            service[g.gate_id] = g.mean_service_seconds
            if g.is_busy:
                remaining = max(self._min_remaining, g.mean_service_seconds - g.busy_seconds)
                avail[g.gate_id] = state.now + remaining
            else:
                avail[g.gate_id] = state.now

        expected_arrivals = state.arrival_rate_per_minute * self._lookahead / 60.0
        n_plan = min(
            self._max_plan_trucks,
            state.global_queue_length + math.ceil(expected_arrivals),
        )

        free_ids = {g.gate_id for g in free}
        used_free: set[str] = set()
        for _ in range(n_plan):
            gid = min(avail, key=lambda k: (avail[k] + service[k], service[k], k))
            if gid in free_ids and avail[gid] == state.now:
                used_free.add(gid)
            avail[gid] += service[gid]

        if not used_free:
            return None    # hold: a busy gate is expected to clear the truck sooner

        chosen_id = min(used_free, key=lambda k: (service[k], k))
        held_back = sorted(free_ids - used_free)
        reason = (
            f"unstrict FIFO v2: expected done in {service[chosen_id]:.0f}s at {chosen_id} "
            f"(plan {n_plan} trucks: {state.global_queue_length} queued + "
            f"{expected_arrivals:.1f} expected"
        )
        if held_back:
            reason += f"; keeping {','.join(held_back)} in reserve"
        reason += ")"
        return Decision(gate_id=chosen_id, reason=reason)


# --- Baselines ---

class StrictFIFORouter(Router):
    """Head-of-queue assigned to whichever gate is free.

    When several gates are free, the tie-break must not depend on gate
    names: breaking ties alphabetically happens to favour slow Gate A in
    the evaluation config, which would make this baseline look worse than
    a real strict-FIFO depot. Pass an `rng` to break ties uniformly at
    random (what the evaluation does). Without one, ties fall back to
    gate_id order, which keeps unit tests deterministic.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng

    @property
    def name(self) -> str:
        return "StrictFIFO"

    def choose(self, state: RouterState) -> Optional[Decision]:
        if state.global_queue_length == 0:
            return None
        free = _free_gates(state)
        if not free:
            return None
        if self._rng is None:
            chosen = min(free, key=lambda g: g.gate_id)
        else:
            chosen = self._rng.choice(sorted(free, key=lambda g: g.gate_id))
        return Decision(gate_id=chosen.gate_id, reason="strict FIFO: next free gate")


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
    """Send to the gate with the fewest waiting trucks. Ties broken by gate_id.

    Only meaningful with per-gate queues. Under the simulator's single global
    queue every gate reports queue_length=0, so this degenerates to
    StrictFIFO's alphabetical tie-break and produces identical results; it
    is therefore excluded from the evaluation roster.
    """

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