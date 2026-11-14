"""One vehicle profile per truck pass (FR5, milestone M8).

A profile joins what the separate readers found for one checkpoint event:

    container codes   src.fusion  (MultiContainerFuser + resolve_shared_codes)
    size/type         voted with the codes (22G1, 45G1, ...)
    vehicle type      src.vehicle_type (fused over cameras)
    vehicle plate     src.plates  (prime mover)
    trailer plate     src.plates  ("T/..." on the trailer rear)
    make              badge text (filled in later; None until then)
    capacity class    derived here, for the router (FR7)

The record mirrors the "Collection" box of the haulier's consignment
note (container no., size/type, prime mover no., trailer no., time), so a
profile can later be matched against expected arrivals.

Capacity class rules (decided for M8, 2026-11-14; the brief left them
open). They are about how long a truck occupies a gate, so they follow
the load, not the paperwork:

    light_vehicle                                  -> light
    box_truck                                      -> medium
    container carrier, no container read           -> medium   (bobtail / empty chassis)
    container carrier, one 10 or 20 ft container   -> medium
    container carrier, two containers or 40/45 ft  -> heavy
    tanker                                         -> heavy
    container carrier, container but size unread   -> unknown
    other / unknown type, no container read        -> unknown

A read container code means the vehicle carries a container whatever the
classifier said, so an "unknown" or "other" vehicle with a code is
treated as a container carrier (the classifier is the weaker evidence).
"No container read" on a container carrier is not proof of an empty
chassis — a code can be missed — which is why it maps to medium rather
than light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .fusion import FusedRead
from .plates import FusedPlate
from .vehicle_type import UNKNOWN, FusedType

LIGHT = "light"
MEDIUM = "medium"
HEAVY = "heavy"
UNKNOWN_CLASS = "unknown"
CAPACITY_CLASSES = (LIGHT, MEDIUM, HEAVY, UNKNOWN_CLASS)


def capacity_class(vehicle_type: str, codes: tuple[str, ...], size_type: Optional[str]) -> str:
    """Capacity class from vehicle type and load (rules in the module docstring)."""
    if codes and vehicle_type in (UNKNOWN, "other"):
        vehicle_type = "container_carrier"
    if vehicle_type == "light_vehicle":
        return LIGHT
    if vehicle_type == "box_truck":
        return MEDIUM
    if vehicle_type == "tanker":
        return HEAVY
    if vehicle_type == "container_carrier":
        if not codes:
            return MEDIUM
        if len(codes) > 1:
            return HEAVY
        if not size_type:
            return UNKNOWN_CLASS
        return MEDIUM if size_type[0] in "12" else HEAVY        # 10 or 20 ft
    return UNKNOWN_CLASS


@dataclass
class VehicleProfile:
    """Everything known about one truck pass."""
    pass_uid: str                         # src.live_events.event_id
    entry_time: datetime
    end_time: datetime
    cameras: tuple[str, ...]
    codes: tuple[str, ...] = ()           # 0-2 container codes, main code first
    code_status: str = "failed"           # verified | recovered | inferred | failed
    code_support: int = 0                 # frames that read the main code
    size_type: Optional[str] = None
    vehicle_type: str = UNKNOWN
    vehicle_type_conf: float = 0.0
    plate: Optional[str] = None
    plate_support: int = 0
    trailer_plate: Optional[str] = None
    trailer_plate_support: int = 0
    make: Optional[str] = None
    capacity_class: str = UNKNOWN_CLASS
    notes: list[str] = field(default_factory=list)

    @property
    def code(self) -> Optional[str]:
        return self.codes[0] if self.codes else None


def build_profile(pass_uid: str, entry_time: datetime, end_time: datetime,
                  cameras: tuple[str, ...], *, code: Optional[FusedRead] = None,
                  vtype: Optional[FusedType] = None, plate: Optional[FusedPlate] = None,
                  trailer: Optional[FusedPlate] = None, make: Optional[str] = None) -> VehicleProfile:
    """Join the fused readings of one pass; any of them may be missing."""
    code = code or FusedRead(None, "failed")
    vtype = vtype or FusedType(UNKNOWN)
    notes = []
    if code.note:
        notes.append(f"code: {code.note}")
    if plate is not None and plate.note:
        notes.append(f"plate: {plate.note}")
    if trailer is not None and trailer.note:
        notes.append(f"trailer plate: {trailer.note}")
    codes = tuple(code.codes)
    return VehicleProfile(
        pass_uid=pass_uid, entry_time=entry_time, end_time=end_time, cameras=tuple(cameras),
        codes=codes, code_status=code.status, code_support=code.support,
        size_type=code.size_type, vehicle_type=vtype.vehicle_type,
        vehicle_type_conf=round(float(vtype.conf), 3),
        plate=plate.plate if plate else None, plate_support=plate.support if plate else 0,
        trailer_plate=trailer.plate if trailer else None,
        trailer_plate_support=trailer.support if trailer else 0,
        make=make, capacity_class=capacity_class(vtype.vehicle_type, codes, code.size_type),
        notes=notes,
    )


def refresh_codes(profile: VehicleProfile, code: FusedRead) -> bool:
    """Update a profile after resolve_shared_codes changed its codes; True if it changed."""
    codes = tuple(code.codes)
    if codes == profile.codes:
        return False
    profile.codes = codes
    profile.notes = [n for n in profile.notes if not n.startswith("code: ")]
    if code.note:
        profile.notes.insert(0, f"code: {code.note}")
    profile.capacity_class = capacity_class(profile.vehicle_type, codes, profile.size_type)
    return True
