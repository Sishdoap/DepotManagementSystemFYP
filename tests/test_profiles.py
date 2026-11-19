"""Tests for vehicle profiles and capacity classes (src/profiles.py)."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.fusion import FusedRead
from src.plates import FusedPlate
from src.profiles import (HEAVY, LIGHT, MEDIUM, UNKNOWN_CLASS, build_profile, capacity_class,
                          refresh_codes)
from src.vehicle_type import UNKNOWN, FusedType

A, C = "EMCU6461774", "MSKU3398188"
T0 = datetime(2026, 11, 13, 11, 39, 2)


@pytest.mark.parametrize("vtype, codes, size, expected", [
    ("light_vehicle", (), None, LIGHT),
    ("box_truck", (), None, MEDIUM),
    ("tanker", (), None, HEAVY),
    ("container_carrier", (), None, MEDIUM),           # bobtail / empty chassis
    ("container_carrier", (A,), "22G1", MEDIUM),
    ("container_carrier", (A,), "45G1", HEAVY),
    ("container_carrier", (A,), "10S2", MEDIUM),           # 10 ft
    ("container_carrier", (A, C), "22G1", HEAVY),      # two 20 ft containers
    ("container_carrier", (A,), None, UNKNOWN_CLASS),  # container, size not read
    (UNKNOWN, (A,), "45G1", HEAVY),                    # a code means a container carrier
    ("other", (A,), "22G1", MEDIUM),
    (UNKNOWN, (), None, UNKNOWN_CLASS),
    ("other", (), None, UNKNOWN_CLASS),
])
def test_capacity_class(vtype, codes, size, expected):
    assert capacity_class(vtype, codes, size) == expected


def test_build_profile_joins_readings():
    code = FusedRead(A, "verified", support=12, size_type="45G1", note="also read: X")
    p = build_profile("u1", T0, T0 + timedelta(seconds=40), ("561", "557"), code=code,
                      vtype=FusedType("container_carrier", 0.93),
                      plate=FusedPlate("BQB7404", 5), trailer=FusedPlate("T/BF7421", 3))
    assert p.code == A and p.codes == (A,) and p.code_status == "verified"
    assert p.size_type == "45G1" and p.capacity_class == HEAVY
    assert p.plate == "BQB7404" and p.plate_support == 5
    assert p.trailer_plate == "T/BF7421" and p.trailer_plate_support == 3
    assert p.vehicle_type == "container_carrier" and p.vehicle_type_conf == 0.93
    assert p.notes == ["code: also read: X"]
    assert p.make is None


def test_build_profile_with_nothing_read():
    p = build_profile("u2", T0, T0, ("561",))
    assert p.codes == () and p.code is None and p.code_status == "failed"
    assert p.vehicle_type == UNKNOWN and p.capacity_class == UNKNOWN_CLASS
    assert p.plate is None and p.trailer_plate is None


def test_refresh_codes_after_shared_code_resolution():
    code = FusedRead(A, "verified", size_type="22G1", codes=(A, C))
    p = build_profile("u3", T0, T0, ("557",), code=code, vtype=FusedType("container_carrier", 0.9))
    assert p.capacity_class == HEAVY
    code.codes, code.note = (A,), "dropped second container MSKU3398188 (main code of event 9)"
    assert refresh_codes(p, code)
    assert p.codes == (A,) and p.capacity_class == MEDIUM
    assert p.notes[0].startswith("code: dropped second container")
    assert not refresh_codes(p, code)
