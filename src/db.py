"""Database layer for the depot simulator.

Uses SQLAlchemy Core for portability between SQLite (FYP1) and PostgreSQL (FYP2).
The connection URL determines the backend; queries are identical across both.

Schema:
    events                — every truck arrival/departure cycle
    containers            — ISO 6346 code reads (1:1 nullable with events)
    vehicle_profiles      — MMR + ALPR results (1:1 nullable with events; FYP2)
    gate_throughput       — per-gate per-minute aggregates for the router
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError


# --- Schema definition ---

metadata = MetaData()

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("plate", String(32), nullable=True),          # may be unknown at arrival
    Column("arrived_at", Float, nullable=False),         # unix timestamp
    Column("assigned_gate", String(8), nullable=True),
    Column("assigned_at", Float, nullable=True),
    Column("departed_at", Float, nullable=True),
    Column("status", String(16), nullable=False),        # waiting | at_gate | done | error
    Column("routing_reason", String(256), nullable=True),  # why this gate was chosen
    Index("ix_events_status", "status"),
    Index("ix_events_arrived_at", "arrived_at"),
)

containers = Table(
    "containers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", Integer, ForeignKey("events.id"), nullable=False),
    Column("raw_code", String(16), nullable=True),       # OCR's argmax read
    Column("recovered_code", String(16), nullable=True), # post-recovery, validated
    Column("is_valid", Integer, nullable=False),         # 1 if validated, 0 otherwise
    Column("recovery_edits", Integer, nullable=True),    # number of recovered chars
    Column("log_probability", Float, nullable=True),     # confidence proxy
    Column("recorded_at", Float, nullable=False),
    Index("ix_containers_event_id", "event_id"),
    Index("ix_containers_recovered_code", "recovered_code"),
)

vehicle_profiles = Table(
    "vehicle_profiles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", Integer, ForeignKey("events.id"), nullable=False),
    # FYP2 fields — nullable in FYP1 so the table exists but rows omit them
    Column("vehicle_class", String(32), nullable=True),  # car|truck|forklift
    Column("make", String(64), nullable=True),
    Column("model", String(64), nullable=True),
    Column("plate_recognized", String(32), nullable=True),
    Column("plate_confidence", Float, nullable=True),
    Column("recorded_at", Float, nullable=False),
    Index("ix_vehicle_profiles_event_id", "event_id"),
)

gate_throughput = Table(
    "gate_throughput",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("gate_id", String(8), nullable=False),
    Column("window_start", Float, nullable=False),       # unix ts of minute bucket
    Column("trucks_served", Integer, nullable=False, default=0),
    Column("mean_service_time", Float, nullable=True),
    Index("ix_throughput_gate_window", "gate_id", "window_start", unique=True),
)

sim_state = Table(
    "sim_state",
    metadata,
    Column("key", String(32), primary_key=True),
    Column("value", Float, nullable=False),
)


# --- Connection management ---

DEFAULT_URL = "sqlite:///depot.db"


def make_engine(url: str | None = None, echo: bool = False) -> Engine:
    """Create a database engine.

    Args:
        url: SQLAlchemy connection URL. Defaults to env var DEPOT_DB_URL,
            then to DEFAULT_URL. Examples:
                sqlite:///depot.db
                postgresql+psycopg://user:pw@host/dbname
        echo: log SQL to stderr (useful for debugging).

    Notes:
        For SQLite we enable WAL journal mode so a reader (dashboard) doesn't
        block a writer (simulator). check_same_thread=False allows the same
        engine to be used from multiple threads — SQLAlchemy handles pooling.
    """
    url = url or os.environ.get("DEPOT_DB_URL") or DEFAULT_URL

    connect_args: dict = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        echo=echo,
        connect_args=connect_args,
        future=True,
    )

    if url.startswith("sqlite"):
        # WAL mode for concurrent reads — set once per connection.
        with engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")

    return engine


def init_schema(engine: Engine) -> None:
    """Create all tables if they don't exist. Idempotent."""
    metadata.create_all(engine)


def drop_schema(engine: Engine) -> None:
    """Drop all tables. Useful for tests."""
    metadata.drop_all(engine)


@contextmanager
def transaction(engine: Engine) -> Iterator:
    """Context manager for a single transaction.

    Usage:
        with transaction(engine) as conn:
            conn.execute(events.insert().values(...))
    """
    with engine.begin() as conn:
        yield conn


# --- Domain operations ---

@dataclass
class EventRow:
    """Plain data class for an event record (returned by query helpers)."""
    id: int
    plate: str | None
    arrived_at: float
    assigned_gate: str | None
    assigned_at: float | None
    departed_at: float | None
    status: str
    routing_reason: str | None


def insert_event(
    engine: Engine,
    *,
    plate: str | None,
    arrived_at: float,
    status: str = "waiting",
) -> int:
    """Insert a new waiting event. Returns the event id."""
    with transaction(engine) as conn:
        result = conn.execute(
            events.insert().values(
                plate=plate,
                arrived_at=arrived_at,
                status=status,
            )
        )
        return int(result.inserted_primary_key[0])


def update_event_assignment(
    engine: Engine,
    event_id: int,
    *,
    gate: str,
    assigned_at: float,
    routing_reason: str | None = None,
) -> None:
    """Mark an event as assigned to a gate."""
    with transaction(engine) as conn:
        conn.execute(
            events.update()
            .where(events.c.id == event_id)
            .values(
                assigned_gate=gate,
                assigned_at=assigned_at,
                status="at_gate",
                routing_reason=routing_reason,
            )
        )


def complete_event(engine: Engine, event_id: int, departed_at: float) -> None:
    """Mark an event as completed."""
    with transaction(engine) as conn:
        conn.execute(
            events.update()
            .where(events.c.id == event_id)
            .values(departed_at=departed_at, status="done")
        )


def insert_container_reading(
    engine: Engine,
    *,
    event_id: int,
    raw_code: str | None,
    recovered_code: str | None,
    is_valid: bool,
    recovery_edits: int | None,
    log_probability: float | None,
    recorded_at: float,
) -> int:
    """Record an ISO 6346 OCR reading (raw + recovered)."""
    with transaction(engine) as conn:
        result = conn.execute(
            containers.insert().values(
                event_id=event_id,
                raw_code=raw_code,
                recovered_code=recovered_code,
                is_valid=1 if is_valid else 0,
                recovery_edits=recovery_edits,
                log_probability=log_probability,
                recorded_at=recorded_at,
            )
        )
        return int(result.inserted_primary_key[0])


def insert_vehicle_profile(
    engine: Engine,
    *,
    event_id: int,
    vehicle_class: str | None = None,
    make: str | None = None,
    model: str | None = None,
    plate_recognized: str | None = None,
    plate_confidence: float | None = None,
    recorded_at: float,
) -> int:
    """Insert a profile-fusion result. All fields nullable for FYP1."""
    with transaction(engine) as conn:
        result = conn.execute(
            vehicle_profiles.insert().values(
                event_id=event_id,
                vehicle_class=vehicle_class,
                make=make,
                model=model,
                plate_recognized=plate_recognized,
                plate_confidence=plate_confidence,
                recorded_at=recorded_at,
            )
        )
        return int(result.inserted_primary_key[0])


# --- Throughput tracking (feeds the unstrict FIFO router) ---

def record_throughput_sample(
    engine: Engine,
    *,
    gate_id: str,
    completed_at: float,
    service_time: float,
) -> None:
    """Increment the per-minute throughput bucket for a gate.

    Service time is the gate-occupancy duration (assigned_at to departed_at)
    for that truck. We bucket by the wall-clock minute at completion.
    """
    bucket = int(completed_at // 60) * 60     # round down to minute boundary
    with transaction(engine) as conn:
        # Try update first; if no row exists, insert.
        existing = conn.execute(
            select(gate_throughput.c.id, gate_throughput.c.trucks_served,
                   gate_throughput.c.mean_service_time)
            .where(gate_throughput.c.gate_id == gate_id)
            .where(gate_throughput.c.window_start == bucket)
        ).first()

        if existing is None:
            conn.execute(
                gate_throughput.insert().values(
                    gate_id=gate_id,
                    window_start=bucket,
                    trucks_served=1,
                    mean_service_time=service_time,
                )
            )
        else:
            new_count = existing.trucks_served + 1
            # Running average: mean_new = mean_old + (x - mean_old) / n_new
            old_mean = existing.mean_service_time or service_time
            new_mean = old_mean + (service_time - old_mean) / new_count
            conn.execute(
                gate_throughput.update()
                .where(gate_throughput.c.id == existing.id)
                .values(trucks_served=new_count, mean_service_time=new_mean)
            )


def recent_throughput(
    engine: Engine,
    gate_id: str,
    *,
    now: float,
    window_seconds: int = 600,
) -> float:
    """Trucks-per-minute for `gate_id` over the last `window_seconds` seconds.

    Used by the unstrict FIFO router to score gates.
    Returns 0.0 if no completions in the window.
    """
    cutoff = now - window_seconds
    with transaction(engine) as conn:
        result = conn.execute(
            select(func.coalesce(func.sum(gate_throughput.c.trucks_served), 0))
            .where(gate_throughput.c.gate_id == gate_id)
            .where(gate_throughput.c.window_start >= cutoff)
        ).scalar()
    trucks = float(result or 0)
    return trucks / (window_seconds / 60.0)

def update_sim_clock(engine: Engine, sim_now: float) -> None:
    """Update the simulator's current time so the dashboard can compute waits."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    with transaction(engine) as conn:
        stmt = sqlite_insert(sim_state).values(key="sim_now", value=sim_now)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"value": sim_now}
        )
        conn.execute(stmt)


def get_sim_clock(engine: Engine) -> float | None:
    """Read the last-recorded simulator time. Returns None if not set."""
    with transaction(engine) as conn:
        row = conn.execute(
            select(sim_state.c.value).where(sim_state.c.key == "sim_now")
        ).first()
    return float(row.value) if row else None


# --- Query helpers (for the dashboard) ---

def list_waiting(engine: Engine) -> list[EventRow]:
    """All trucks currently waiting for a gate, oldest first."""
    with transaction(engine) as conn:
        rows = conn.execute(
            select(events).where(events.c.status == "waiting").order_by(events.c.arrived_at)
        ).all()
    return [_row_to_event(r) for r in rows]


def list_at_gate(engine: Engine) -> list[EventRow]:
    """All trucks currently being served at a gate."""
    with transaction(engine) as conn:
        rows = conn.execute(
            select(events).where(events.c.status == "at_gate").order_by(events.c.assigned_at)
        ).all()
    return [_row_to_event(r) for r in rows]


def list_recent_containers(engine: Engine, limit: int = 30) -> list[dict]:
    """Latest container readings with their associated event info."""
    with transaction(engine) as conn:
        rows = conn.execute(
            select(
                containers.c.recorded_at,
                containers.c.raw_code,
                containers.c.recovered_code,
                containers.c.is_valid,
                events.c.plate,
                events.c.assigned_gate,
            )
            .select_from(containers.join(events, containers.c.event_id == events.c.id))
            .order_by(containers.c.recorded_at.desc())
            .limit(limit)
        ).all()
    return [dict(r._mapping) for r in rows]


def _row_to_event(row) -> EventRow:
    return EventRow(
        id=row.id,
        plate=row.plate,
        arrived_at=row.arrived_at,
        assigned_gate=row.assigned_gate,
        assigned_at=row.assigned_at,
        departed_at=row.departed_at,
        status=row.status,
        routing_reason=row.routing_reason,
    )