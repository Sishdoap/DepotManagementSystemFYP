"""Discrete-event simulation harness for the depot traffic management system.

Drives arrivals, dispatching, gate service, OCR, and database writes as a
coherent system using SimPy. Two execution modes share one core:

    fast_mode=True    Uses simpy.Environment (no real-time delay). For
                      multi-seed evaluation runs (~30 seeds in seconds).
    fast_mode=False   Uses simpy.RealtimeEnvironment (wall-clock paced).
                      For the dashboard demo, where you want to see trucks
                      move through the system over real time.

Both modes use the same Router, OCR adapter, and database — only the
SimPy environment differs. This satisfies your NFR for reproducibility
(fixed seed -> identical fast-mode results) and serves the dashboard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import simpy

from .image_source import ImageSource

from .db import (
    complete_event,
    insert_container_reading,
    insert_event,
    record_throughput_sample,
    recent_throughput,
    update_event_assignment,
)
from .ocr import OCRAdapter, OCRResult
from .router import (
    Decision,
    GateState,
    Router,
    RouterState,
)
from .synthetic_images import ContainerImageGenerator, GenerationConfig


# --- Configuration ---

@dataclass(frozen=True)
class GateConfig:
    """Per-gate service-time distribution.

    Service time is sampled as max(0.1, normal(mean_seconds, std_seconds))
    per truck. Different means per gate are what make routing decisions
    matter — this is the heterogeneity unstrict FIFO is supposed to exploit.
    """
    gate_id: str
    mean_service_seconds: float
    std_service_seconds: float


@dataclass(frozen=True)
class SimulationConfig:
    """All knobs for one simulation run."""
    duration_seconds: float = 3600.0            # 1 hour of simulated time
    arrival_rate_per_minute: float = 1.5        # Poisson arrival rate
    gates: tuple[GateConfig, ...] = (
        GateConfig("A", mean_service_seconds=90.0, std_service_seconds=15.0),
        GateConfig("B", mean_service_seconds=60.0, std_service_seconds=10.0),
        GateConfig("C", mean_service_seconds=45.0, std_service_seconds=8.0),
    )
    image_config: GenerationConfig = field(default_factory=GenerationConfig)
    throughput_window_seconds: int = 600        # rolling window for router
    seed: int = 0                                # for reproducibility


# --- Per-truck record (in-memory) ---

@dataclass
class Truck:
    """Truck object passed through the simulation."""
    truck_id: int
    arrived_at: float
    plate: str
    event_id: Optional[int] = None              # set after DB insert
    assigned_gate: Optional[str] = None
    assigned_at: Optional[float] = None
    departed_at: Optional[float] = None


# --- Results ---

@dataclass
class SimulationResult:
    """What one simulation run produces. The DB has the full record;
    this is a lightweight summary for the evaluation harness."""
    router_name: str
    seed: int
    n_arrivals: int
    n_completed: int
    n_recoveries: int            # trucks whose OCR was recovered (raw != recovered)
    n_ocr_valid: int             # trucks whose container code came out valid
    wait_times: list[float]      # per-truck wait time (assigned_at - arrived_at)
    service_times: list[float]   # per-truck service time (departed_at - assigned_at)
    per_gate_utilization: dict[str, float]
    per_gate_throughput: dict[str, int]


# --- The simulation itself ---

class DepotSimulation:
    """Runs one simulation. Use `run()` once; create a new instance per run."""

    def __init__(
        self,
        *,
        config: SimulationConfig,
        router: Router,
        ocr: OCRAdapter,
        engine,                          # SQLAlchemy engine from db.make_engine()
        image_source: "ImageSource",      # CHANGED
        fast_mode: bool = True,
        realtime_factor: float = 1.0,  # only used in real-time mode
        share_frames: bool = False,
    ):
        self.config = config
        self.router = router
        self.ocr = ocr
        self.engine = engine
        self.image_source = image_source
        self.fast_mode = fast_mode
        self.share_frames = share_frames

        # SimPy environment.
        if fast_mode:
            self.env = simpy.Environment()
        else:
            self.env = simpy.RealtimeEnvironment(factor=realtime_factor, strict=False)

        # RNGs — separate streams per concern, all seeded from config.seed
        # so the full run is reproducible.
        self._rng_arrivals = random.Random(config.seed)
        self._rng_service = random.Random(config.seed + 1)
        self._rng_plates = random.Random(config.seed + 2)
        self._np_rng = np.random.default_rng(config.seed + 3)

        # Per-gate state. Each gate is a SimPy Resource with capacity=1.
        self.gates: dict[str, simpy.Resource] = {
            gc.gate_id: simpy.Resource(self.env, capacity=1)
            for gc in config.gates
        }
        self.gate_configs: dict[str, GateConfig] = {gc.gate_id: gc for gc in config.gates}
        self.gate_busy: dict[str, bool] = {gc.gate_id: False for gc in config.gates}
        self.gate_queue_lengths: dict[str, int] = {gc.gate_id: 0 for gc in config.gates}

        # Global queue (head-of-queue dispatched by the router).
        self.waiting: list[Truck] = []

        # Event signaling that the dispatcher should re-evaluate.
        # Triggered when a truck arrives or a gate frees.
        self.dispatch_event: simpy.Event = self.env.event()

        # Stats accumulators.
        self._truck_id_counter = 0
        self._wait_times: list[float] = []
        self._service_times: list[float] = []
        self._gate_busy_time: dict[str, float] = {g: 0.0 for g in self.gates}
        self._gate_completion_count: dict[str, int] = {g: 0 for g in self.gates}
        self._n_arrivals = 0
        self._n_completed = 0
        self._n_recoveries = 0
        self._n_ocr_valid = 0

    # --- Public API ---

    def run(self) -> SimulationResult:
        """Run the simulation to completion and return summary stats."""
        self.env.process(self._arrival_process())
        self.env.process(self._dispatcher_process())
        self.env.process(self._clock_writer_process())
        self.env.run(until=self.config.duration_seconds)
        return self._build_result()

    # --- SimPy processes ---

    def _arrival_process(self):
        """Poisson-distributed arrivals at the configured rate."""
        mean_interarrival_s = 60.0 / self.config.arrival_rate_per_minute
        while True:
            wait = self._rng_arrivals.expovariate(1.0 / mean_interarrival_s)
            yield self.env.timeout(wait)

            self._truck_id_counter += 1
            truck = Truck(
                truck_id=self._truck_id_counter,
                arrived_at=self.env.now,
                plate=f"W{self._rng_plates.randint(1000, 9999)}",
            )

            # Persist arrival.
            truck.event_id = insert_event(
                self.engine,
                plate=truck.plate,
                arrived_at=truck.arrived_at,
                status="waiting",
            )
            self.waiting.append(truck)
            self._n_arrivals += 1

            # Wake the dispatcher.
            self._signal_dispatcher()

    def _dispatcher_process(self):
        """Wakes on every dispatch_event; tries to assign waiting trucks to gates."""
        while True:
            yield self.dispatch_event
            # Replace the event so the next signal can fire.
            self.dispatch_event = self.env.event()

            # Try to dispatch as many trucks as we can right now.
            # The router may return None if no gates are free or queue is empty.
            while self.waiting:
                state = self._snapshot_state()
                decision = self.router.choose(state)
                if decision is None:
                    break
                truck = self.waiting.pop(0)
                self.env.process(self._gate_service_process(truck, decision))

    def _gate_service_process(self, truck: Truck, decision: Decision):
        """One truck's lifecycle at a gate: assignment, OCR, service, departure."""
        gate_id = decision.gate_id
        gate = self.gates[gate_id]

        # Hold the gate's resource. Capacity=1 means at most one truck per gate.
        with gate.request() as req:
            yield req

            # Assignment.
            truck.assigned_gate = gate_id
            truck.assigned_at = self.env.now
            self.gate_busy[gate_id] = True
            self._wait_times.append(truck.assigned_at - truck.arrived_at)

            update_event_assignment(
                self.engine,
                truck.event_id,
                gate=gate_id,
                assigned_at=truck.assigned_at,
                routing_reason=decision.reason,
            )

            sourced = self.image_source.next_image()
            ocr_result: OCRResult = self.ocr.predict(sourced.image)

            if self.share_frames:
                from .dashboard_io import write_gate_frame
                annotation = (
                    f"Gate {gate_id} | {ocr_result.recovered_code or '???'}"
                    f" | valid={ocr_result.is_valid}"
                )
                # Pass CCLN's bbox through to the dashboard for visualization.
                # MockOCRAdapter returns a fake bbox at (10, 10, 200, 40), which
                # would look wrong on real images — guard with the adapter name.
                bbox_tuple = None
                if (
                    ocr_result.bounding_box is not None
                    and "Real" in self.ocr.name
                ):
                    bb = ocr_result.bounding_box
                    bbox_tuple = (bb.x, bb.y, bb.width, bb.height)
                write_gate_frame(
                    gate_id,
                    np.array(sourced.image),
                    annotation=annotation,
                    bounding_box=bbox_tuple,
                )

            insert_container_reading(
                self.engine,
                event_id=truck.event_id,
                raw_code=ocr_result.raw_string,
                recovered_code=ocr_result.recovered_code,
                is_valid=ocr_result.is_valid,
                recovery_edits=ocr_result.recovery_edits,
                log_probability=ocr_result.log_probability,
                recorded_at=self.env.now,
            )
            if ocr_result.is_valid:
                self._n_ocr_valid += 1
            if ocr_result.recovery_edits and ocr_result.recovery_edits > 0:
                self._n_recoveries += 1

            # Service time (this is what makes gates heterogeneous).
            service_time = self._sample_service_time(gate_id)
            yield self.env.timeout(service_time)

            # Departure.
            truck.departed_at = self.env.now
            actual_service = truck.departed_at - truck.assigned_at
            self._service_times.append(actual_service)
            self._gate_busy_time[gate_id] += actual_service
            self._gate_completion_count[gate_id] += 1
            self._n_completed += 1

            complete_event(self.engine, truck.event_id, truck.departed_at)
            record_throughput_sample(
                self.engine,
                gate_id=gate_id,
                completed_at=truck.departed_at,
                service_time=actual_service,
            )

            self.gate_busy[gate_id] = False
            self._signal_dispatcher()

    # --- Helpers ---

    def _signal_dispatcher(self):
        """Wake the dispatcher. Safe to call repeatedly — succeed is idempotent
        on a fresh Event."""
        if not self.dispatch_event.triggered:
            self.dispatch_event.succeed()

    def _snapshot_state(self) -> RouterState:
        """Build a RouterState from current simulation state."""
        gate_states = []
        for gid in sorted(self.gates.keys()):
            throughput = recent_throughput(
                self.engine,
                gid,
                now=self.env.now,
                window_seconds=self.config.throughput_window_seconds,
            )
            gate_states.append(
                GateState(
                    gate_id=gid,
                    is_busy=self.gate_busy[gid],
                    queue_length=0,    # we use a global queue, not per-gate
                    recent_throughput=throughput,
                )
            )

        head_wait = 0.0
        if self.waiting:
            head_wait = self.env.now - self.waiting[0].arrived_at

        return RouterState(
            gates=gate_states,
            global_queue_length=len(self.waiting),
            head_truck_wait_seconds=head_wait,
            now=self.env.now,
        )

    def _sample_service_time(self, gate_id: str) -> float:
        gc = self.gate_configs[gate_id]
        sample = self._rng_service.gauss(gc.mean_service_seconds, gc.std_service_seconds)
        return max(0.1, sample)

    def _build_result(self) -> SimulationResult:
        utilization = {
            gid: (busy / self.config.duration_seconds)
            for gid, busy in self._gate_busy_time.items()
        }
        return SimulationResult(
            router_name=self.router.name,
            seed=self.config.seed,
            n_arrivals=self._n_arrivals,
            n_completed=self._n_completed,
            n_recoveries=self._n_recoveries,
            n_ocr_valid=self._n_ocr_valid,
            wait_times=list(self._wait_times),
            service_times=list(self._service_times),
            per_gate_utilization=utilization,
            per_gate_throughput=dict(self._gate_completion_count),
        )
    
    def _clock_writer_process(self):
        """Periodically write env.now to the database so the dashboard can
        compute correct wait times against simulated time, not wall-clock time."""
        from .db import update_sim_clock
        while True:
            update_sim_clock(self.engine, self.env.now)
            yield self.env.timeout(1.0)   # update once per simulated second