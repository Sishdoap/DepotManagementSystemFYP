"""Tests for live events -> profiles -> database (src/live_pipeline.py)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.db import init_schema, list_passes, make_engine
from src.events import CameraPass, Event
from src.footage import Detection
from src.fusion import FrameRead
from src.live_events import LiveUpdate, event_id
from src.live_ocr import EventReads, SpooledCrop
from src.live_pipeline import PassRecorder
from src.vehicle_type import FrameType

T0 = datetime(2026, 11, 13, 11, 39, 2)
A, C = "EMCU6461774", "MSKU3398188"


@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    init_schema(eng)
    yield eng
    eng.dispose()


def event(tid, offset_s, cams=("561", "557")):
    passes = [CameraPass(cam, tid, T0 + timedelta(seconds=offset_s + i * 5),
                         T0 + timedelta(seconds=offset_s + i * 5 + 20)) for i, cam in enumerate(cams)]
    return Event(passes)


def dets(e, frames=(0, 5, 10)):
    return {(p.camera, p.track_id): [Detection(f * 0.25, f, "truck", 0.9, (0, 0, 800, 300), p.track_id)
                                     for f in frames] for p in e.passes}


def recorder(engine, jobs):
    rec = PassRecorder(engine, jobs.append, source="replay", clock=lambda: 1.0)
    return rec


def code_reads(*codes_sizes, camera="557"):
    reads = EventReads()
    i = 0
    for code, size, n in codes_sizes:
        for _ in range(n):
            reads.code_reads[(camera, i)] = [FrameRead(camera, i, "verified", code, size)]
            i += 1
    return reads


def add_types(rec, e, top="container_carrier"):
    for p in e.passes:
        rec.add_types(p.camera, {(p.track_id, f): FrameType(p.camera, f, {top: 0.9, "other": 0.1})
                                 for f in (0, 5, 10)})


def test_provisional_then_final(engine):
    jobs = []
    rec = recorder(engine, jobs)
    e = event(1, 0)
    add_types(rec, e)
    rec.add_crop(SpooledCrop("557", 1, 5, 1.25, (0, 0, 800, 300), (0, 0), "/x.jpg"))
    rec.add_crop(SpooledCrop("557", 1, 99, 9.0, (0, 0, 800, 300), (0, 0), "/y.jpg"))   # not kept
    rec.on_update(LiveUpdate(provisional=[e], detections=dets(e)))
    rows = list_passes(engine)
    assert len(rows) == 1 and rows[0]["status"] == "provisional"
    assert rows[0]["vehicle_type"] == "container_carrier" and rows[0]["codes"] == ()
    assert jobs == []

    rec.on_update(LiveUpdate(finalized=[e], detections=dets(e)))
    assert len(jobs) == 1 and jobs[0].pass_uid == event_id(e)
    assert [c.frame for c in jobs[0].crops] == [5]           # only crops of kept detections
    assert list_passes(engine)[0]["status"] == "provisional"  # until OCR is back

    p = rec.on_reads(event_id(e), code_reads((A, "45G1", 6)))
    row = list_passes(engine)[0]
    assert row["status"] == "final" and row["codes"] == (A,) and row["size_type"] == "45G1"
    assert row["capacity_class"] == "heavy" and p.code == A


def test_vanished_provisional_event_is_deleted(engine):
    rec = recorder(engine, [])
    e1, e2 = event(1, 0), event(2, 0, cams=("561",))
    rec.on_update(LiveUpdate(provisional=[e1], detections=dets(e1)))
    rec.on_update(LiveUpdate(provisional=[e2], detections=dets(e2)))   # re-assembled
    assert [r["pass_uid"] for r in list_passes(engine)] == [event_id(e2)]


def test_neighbour_rewritten_when_later_result_owns_its_container(engine):
    jobs = []
    rec = recorder(engine, jobs)
    first, second = event(1, 0), event(2, 15)
    rec.on_update(LiveUpdate(finalized=[first, second], detections={**dets(first), **dets(second)}))
    # the second truck's OCR comes back first and lists the first truck's container
    rec.on_reads(event_id(second), code_reads((A, "22G1", 20), (C, "22G1", 6)))
    row = {r["pass_uid"]: r for r in list_passes(engine)}[event_id(second)]
    assert row["codes"] == (A, C) and row["capacity_class"] == "heavy"
    # then the first truck's: C is its main code, so the second truck loses it
    rec.on_reads(event_id(first), code_reads((C, "22G1", 12)))
    rows = {r["pass_uid"]: r for r in list_passes(engine)}
    assert rows[event_id(first)]["codes"] == (C,)
    assert rows[event_id(second)]["codes"] == (A,)
    assert rows[event_id(second)]["capacity_class"] == "medium"
    assert "dropped second container" in rows[event_id(second)]["notes"]


def test_far_apart_trucks_keep_the_same_code(engine):
    rec = recorder(engine, [])
    a, b = event(1, 0), event(2, 3600)
    rec.on_update(LiveUpdate(finalized=[a, b], detections={**dets(a), **dets(b)}))
    rec.on_reads(event_id(a), code_reads((C, "22G1", 10)))
    rec.on_reads(event_id(b), code_reads((A, "22G1", 20), (C, "22G1", 6)))
    rows = {r["pass_uid"]: r for r in list_passes(engine)}
    assert rows[event_id(b)]["codes"] == (A, C)


def test_old_crop_lists_dropped_unless_in_use(engine):
    rec = recorder(engine, [])
    e = event(1, 0)
    rec.on_update(LiveUpdate(provisional=[e], detections=dets(e)))
    used = SpooledCrop("557", 1, 5, 10.0, (0, 0, 800, 300), (0, 0), "/a.jpg")
    stale = SpooledCrop("557", 9, 5, 10.0, (0, 0, 800, 300), (0, 0), "/b.jpg")
    rec.add_crop(used)
    rec.add_crop(stale)
    assert rec.drop_crops_before({"557": 1000.0}, keep_s=600) == [stale]
    assert rec.drop_crops_before({"557": 1000.0}, keep_s=600) == []
