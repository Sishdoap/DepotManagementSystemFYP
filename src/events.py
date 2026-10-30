"""Assemble per-camera truck passes into checkpoint events.

An *event* is one truck passing the entry identification checkpoint
(option B in docs/FYP2_BRIEF.md): the cameras on the entry lane each see
it once or in a few fragments, and an event gathers all of those into one
record, which later becomes one vehicle profile (FR5).

The cameras form a chain along the lane, in the order a truck reaches
them (config roles cab -> codes -> backup, i.e. 561 -> 558 -> 559):

    561 (barrier, cab head-on)  ->  558 (overhead, doors)  ->  559 (further on)

Assembly rule, derived from the tracked footage (2026-10-30):

- Every pass on the *head* camera opens a new event. On 561 the cab fills
  the lane and the tracker gives exactly one track per truck, so the head
  camera is the most reliable truck counter.
- A pass on a later camera joins the *most recent* open event that it
  could belong to: the event started no later than the pass (plus a small
  clock tolerance), no more than `max_lag_s` earlier, and if the event
  already has passes on that camera, the new pass follows them within
  `merge_gap_s`. "Most recent" is right because the entry lane is a single
  lane — trucks cannot overtake, so passes arrive in truck order.
- A later-camera pass that fits no event opens its own event (the head
  camera missed that truck). Such events are flagged by `head_seen`.

Measured on the footage: head -> 558 lag is 2-8 s, up to ~28 s when a
truck stops at the barrier; 558 fragments of one truck (cab first, then
the container doors) are up to ~20 s apart; consecutive trucks start at
least ~30 s apart. The defaults sit between those numbers. They are
footage-specific heuristics, not general truths — revisit them if more
footage arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Optional

from .footage import (
    CAMERA_ROLES, CameraConfig, Detection, RecordingInfo, group_by_track, pass_rejections,
    wall_clock,
)


@dataclass(frozen=True)
class CameraPass:
    """One kept track on one camera, in corrected wall-clock time."""
    camera: str
    track_id: int
    start: datetime
    end: datetime


@dataclass
class Event:
    """One truck through the checkpoint: its passes on every camera."""
    passes: list[CameraPass] = field(default_factory=list)
    head_seen: bool = True       # False if the head camera missed this truck

    @property
    def start(self) -> datetime:
        """Entry timestamp: when the truck first appears on any camera."""
        return min(p.start for p in self.passes)

    @property
    def end(self) -> datetime:
        return max(p.end for p in self.passes)

    @property
    def cameras(self) -> list[str]:
        """Cameras that saw the truck, in first-seen order."""
        seen: list[str] = []
        for p in sorted(self.passes, key=lambda p: p.start):
            if p.camera not in seen:
                seen.append(p.camera)
        return seen

    def on(self, camera: str) -> list[CameraPass]:
        return sorted((p for p in self.passes if p.camera == camera), key=lambda p: p.start)


def assemble_events(
    passes: Iterable[CameraPass],
    chain: list[str],
    *,
    max_lag_s: float = 45.0,
    merge_gap_s: float = 25.0,
    head_merge_gap_s: float = 5.0,
    clock_tolerance_s: float = 2.0,
) -> list[Event]:
    """Group camera passes into events; see the module docstring for the rule.

    chain              camera ids in the order a truck reaches them;
                       chain[0] is the head camera. Passes on cameras not
                       in the chain are ignored.
    max_lag_s          latest a later-camera pass may start after its event.
    merge_gap_s        largest gap between fragments of one truck on one
                       later camera.
    head_merge_gap_s   head-camera tracks this close together are one truck
                       whose track broke (rare on 561, kept small so two
                       real trucks never merge).
    clock_tolerance_s  a pass may start this much *before* its event, to
                       absorb residual clock offset between cameras.
    """
    if not chain:
        raise ValueError("chain must name at least the head camera")
    head = chain[0]
    order = {cam: i for i, cam in enumerate(chain)}
    tol = timedelta(seconds=clock_tolerance_s)
    # Sort by start time, but delay later-camera passes by the clock
    # tolerance: a head pass starting up to `tol` after a later-camera pass
    # must still open its event first, or the later pass would open a
    # spurious event of its own. Ties go to the earlier camera in the chain.
    todo = sorted((p for p in passes if p.camera in order),
                  key=lambda p: (p.start + (tol if p.camera != head else timedelta(0)),
                                 order[p.camera]))

    events: list[Event] = []
    for p in todo:
        if p.camera == head:
            last = events[-1] if events else None
            prev = last.on(head) if last else []
            if prev and (p.start - prev[-1].end).total_seconds() <= head_merge_gap_s:
                last.passes.append(p)
            else:
                events.append(Event([p]))
            continue

        # The most recent event that had begun by now (within clock tolerance).
        # Only that one is a candidate: with no overtaking, a pass can't
        # belong to an older truck once a newer one has entered.
        target: Optional[Event] = next(
            (e for e in reversed(events) if e.start <= p.start + tol), None)
        if target is not None:
            same_cam = target.on(p.camera)
            too_late = (p.start - target.start).total_seconds() > max_lag_s
            if same_cam:
                # A continuing fragment may run past max_lag_s (a truck that
                # stops under the camera); it only has to follow closely.
                too_late = (p.start - same_cam[-1].end).total_seconds() > merge_gap_s
            if too_late:
                target = None
        if target is None:
            events.append(Event([p], head_seen=False))
        else:
            target.passes.append(p)

    return sorted(events, key=lambda e: e.start)


def camera_chain(configs: Mapping[str, CameraConfig]) -> list[str]:
    """Cameras in the order a truck reaches them: roles cab -> codes -> backup."""
    return sorted((c for c, cfg in configs.items() if cfg.role != "ignore"),
                  key=lambda c: CAMERA_ROLES.index(configs[c].role))


@dataclass
class ExtractedEvents:
    """Everything scripts need from one extraction run."""
    chain: list[str]
    events: list[Event]
    detections: dict[tuple[str, int], list[Detection]]   # (camera, track id) -> kept detections
    rejected: list[dict]                                   # one row per dropped track


def extract_events(
    tracks: Mapping[str, list[Detection]],
    configs: Mapping[str, CameraConfig],
    recordings: Mapping[str, RecordingInfo],
    **assembly_kwargs,
) -> ExtractedEvents:
    """Tracked detections -> lane-filtered, clock-corrected passes -> events.

    Shared by scripts/extract_events.py and the scripts that read the
    events (M2 onward), so event numbering is identical everywhere: events
    are numbered 1.. in entry-time order, and the same tracks + config give
    the same events.
    """
    chain = camera_chain(configs)
    passes, detections, rejected = [], {}, []
    for cam in chain:
        cfg, info = configs[cam], recordings[cam]
        for p in group_by_track(tracks.get(cam, [])):
            tid = p.detections[0].track_id
            start = wall_clock(info, p.start, cfg.offset_at(p.start))
            end = wall_clock(info, p.end, cfg.offset_at(p.end))
            reasons = pass_rejections(p, cfg)
            if reasons:
                rejected.append({"camera": cam, "track_id": tid, "start": f"{start:%H:%M:%S}",
                                 "end": f"{end:%H:%M:%S}", "frames": len(p.detections),
                                 "reason": "; ".join(reasons)})
            else:
                passes.append(CameraPass(cam, tid, start, end))
                detections[(cam, tid)] = p.detections
    events = assemble_events(passes, chain, **assembly_kwargs)
    return ExtractedEvents(chain, events, detections, rejected)
