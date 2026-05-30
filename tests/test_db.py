"""Tests for the database layer.

Every test runs against a fresh in-memory SQLite database, so they're fast
(< 1s each) and fully isolated from each other.
"""

import time
import pytest
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.db import (
    EventRow,
    complete_event,
    drop_schema,
    init_schema,
    insert_container_reading,
    insert_event,
    insert_vehicle_profile,
    list_at_gate,
    list_recent_containers,
    list_waiting,
    make_engine,
    recent_throughput,
    record_throughput_sample,
    update_event_assignment,
)


@pytest.fixture
def engine():
    """Fresh in-memory SQLite engine per test."""
    eng = make_engine("sqlite:///:memory:")
    init_schema(eng)
    yield eng
    eng.dispose()


class TestEventLifecycle:
    def test_insert_returns_id(self, engine):
        eid = insert_event(engine, plate="WX1234", arrived_at=1000.0)
        assert isinstance(eid, int)
        assert eid > 0

    def test_waiting_then_at_gate_then_done(self, engine):
        eid = insert_event(engine, plate="WX1234", arrived_at=1000.0)
        assert len(list_waiting(engine)) == 1
        assert len(list_at_gate(engine)) == 0

        update_event_assignment(
            engine, eid, gate="A", assigned_at=1005.0,
            routing_reason="highest score (alpha=0.7)",
        )
        assert len(list_waiting(engine)) == 0
        assert len(list_at_gate(engine)) == 1

        complete_event(engine, eid, departed_at=1020.0)
        assert len(list_waiting(engine)) == 0
        assert len(list_at_gate(engine)) == 0

    def test_routing_reason_persisted(self, engine):
        eid = insert_event(engine, plate=None, arrived_at=1000.0)
        update_event_assignment(
            engine, eid, gate="C", assigned_at=1001.0,
            routing_reason="bootstrap: shortest queue",
        )
        rows = list_at_gate(engine)
        assert rows[0].routing_reason == "bootstrap: shortest queue"

    def test_waiting_ordered_by_arrival(self, engine):
        ids = []
        for t in [1003.0, 1001.0, 1002.0]:
            ids.append(insert_event(engine, plate=f"P{t}", arrived_at=t))
        waiting = list_waiting(engine)
        assert [w.arrived_at for w in waiting] == [1001.0, 1002.0, 1003.0]


class TestContainerReadings:
    def test_insert_valid_reading(self, engine):
        eid = insert_event(engine, plate="WX1", arrived_at=1000.0)
        cid = insert_container_reading(
            engine,
            event_id=eid,
            raw_code="CSQU3054383",
            recovered_code="CSQU3054383",
            is_valid=True,
            recovery_edits=0,
            log_probability=-0.5,
            recorded_at=1010.0,
        )
        assert cid > 0

    def test_insert_recovered_reading(self, engine):
        eid = insert_event(engine, plate="WX1", arrived_at=1000.0)
        insert_container_reading(
            engine,
            event_id=eid,
            raw_code="CSQU3054384",     # OCR mis-read
            recovered_code="CSQU3054383",  # corrected
            is_valid=True,
            recovery_edits=1,
            log_probability=-7.5,
            recorded_at=1010.0,
        )
        readings = list_recent_containers(engine)
        assert len(readings) == 1
        assert readings[0]["raw_code"] == "CSQU3054384"
        assert readings[0]["recovered_code"] == "CSQU3054383"

    def test_insert_unrecoverable_reading(self, engine):
        # OCR failed entirely — recovered_code is None, is_valid is False.
        eid = insert_event(engine, plate="WX1", arrived_at=1000.0)
        insert_container_reading(
            engine,
            event_id=eid,
            raw_code="????1234567",
            recovered_code=None,
            is_valid=False,
            recovery_edits=None,
            log_probability=None,
            recorded_at=1010.0,
        )
        readings = list_recent_containers(engine)
        assert readings[0]["is_valid"] == 0


class TestVehicleProfile:
    def test_fyp1_profile_with_only_required_fields(self, engine):
        # In FYP1, MMR/ALPR aren't run, so most fields are None.
        eid = insert_event(engine, plate="WX1", arrived_at=1000.0)
        pid = insert_vehicle_profile(
            engine, event_id=eid, recorded_at=1010.0,
        )
        assert pid > 0

    def test_fyp2_profile_with_all_fields(self, engine):
        eid = insert_event(engine, plate=None, arrived_at=1000.0)
        insert_vehicle_profile(
            engine,
            event_id=eid,
            vehicle_class="truck",
            make="Volvo",
            model="FH16",
            plate_recognized="WX1234",
            plate_confidence=0.93,
            recorded_at=1010.0,
        )


class TestThroughput:
    def test_empty_window_returns_zero(self, engine):
        rate = recent_throughput(engine, "A", now=2000.0, window_seconds=600)
        assert rate == 0.0

    def test_single_completion(self, engine):
        record_throughput_sample(engine, gate_id="A", completed_at=1000.0, service_time=15.0)
        # window_seconds=600 covers 1000.0 from now=1500.0
        rate = recent_throughput(engine, "A", now=1500.0, window_seconds=600)
        # 1 truck over 10 minutes = 0.1 trucks/minute
        assert rate == pytest.approx(0.1, rel=0.01)

    def test_multiple_completions_same_bucket(self, engine):
        # Three trucks finish at 1000, 1010, 1050 — all within the same minute bucket
        for t in [1000.0, 1010.0, 1050.0]:
            record_throughput_sample(engine, gate_id="A", completed_at=t, service_time=10.0)
        rate = recent_throughput(engine, "A", now=1500.0, window_seconds=600)
        assert rate == pytest.approx(0.3, rel=0.01)

    def test_completions_outside_window_excluded(self, engine):
        # 1 truck recently, 1 truck long ago.
        record_throughput_sample(engine, gate_id="A", completed_at=500.0, service_time=10.0)
        record_throughput_sample(engine, gate_id="A", completed_at=1400.0, service_time=10.0)
        rate = recent_throughput(engine, "A", now=1500.0, window_seconds=600)
        # Only the second truck counts (500 is > 600s before now).
        assert rate == pytest.approx(0.1, rel=0.01)

    def test_per_gate_isolation(self, engine):
        record_throughput_sample(engine, gate_id="A", completed_at=1400.0, service_time=10.0)
        record_throughput_sample(engine, gate_id="B", completed_at=1400.0, service_time=10.0)
        record_throughput_sample(engine, gate_id="B", completed_at=1410.0, service_time=10.0)
        assert recent_throughput(engine, "A", now=1500.0, window_seconds=600) == pytest.approx(0.1)
        assert recent_throughput(engine, "B", now=1500.0, window_seconds=600) == pytest.approx(0.2)


class TestSchemaPortability:
    def test_drop_and_recreate_idempotent(self, engine):
        # Insert data, drop, recreate, expect empty.
        insert_event(engine, plate="WX1", arrived_at=1000.0)
        assert len(list_waiting(engine)) == 1

        drop_schema(engine)
        init_schema(engine)
        assert len(list_waiting(engine)) == 0

    def test_init_schema_idempotent(self, engine):
        # Calling init_schema twice should not error or duplicate.
        init_schema(engine)
        init_schema(engine)
        # If we got here, no exception.