"""Assemble per-camera truck passes into checkpoint events.

An *event* is one truck passing the entry identification checkpoint
(option B in docs/FYP2_BRIEF.md): the cameras on the entry lane each see
it once or in a few fragments, and an event gathers all of those into one
record, which later becomes one vehicle profile (FR5).

The cameras form a chain along the lane (config roles cab -> side ->
codes -> backup, i.e. 561 -> 557/560 -> 558 -> 559). The head camera,
561 at the barrier, sees the cab head-on; the side cameras 557/560 see
the truck side-on; 558 sees its doors as it drives away; 559 is further
on.

Each pass is compared with the head camera through its *reference time*:
its start, or — for cameras configured with `anchor: end` — its end. The
side cameras need the end: they see a truck while it waits at the edge
of their frame, sometimes a minute before it reaches the barrier, but the
pass ends a few seconds after the truck passes 561.

Assembly rule, derived from the tracked footage (2026-10-30, side
cameras added 2026-11-02):

- Every pass on the *head* camera opens a new event. On 561 the cab fills
  the lane and the tracker gives exactly one track per truck, so the head
  camera is the most reliable truck counter. The event's *entry time*
  (checkpoint time, used for numbering) is that pass's start.
- A pass on a later camera joins the *most recent* event whose entry
  time is no later than the pass's reference time (plus a small clock
  tolerance), if the reference time is at most `max_lag_s` after it — or,
  if the event already has passes on that camera, if the new pass starts
  within `merge_gap_s` of the previous one ending. "Most recent" is right
  because the entry lane is a single lane — trucks cannot overtake, so
  passes arrive in truck order.
- A later-camera pass that fits no event opens its own event (the head
  camera missed that truck), flagged by `head_seen`; its entry time is
  that pass's reference time.

Each event also records `first_seen` (earliest pass start on any camera),
which includes time spent waiting in view of the side cameras.

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
    anchor_end: bool = False     # compare this pass by its end (side cameras)

    @property
    def ref(self) -> datetime:
        """Reference time used to match the pass to an event."""
        return self.end if self.anchor_end else self.start


@dataclass
class Event:
    """One truck through the checkpoint: its passes on every camera."""
    passes: list[CameraPass] = field(default_factory=list)
    head_seen: bool = True       # False if the head camera missed this truck
    entry_time: Optional[datetime] = None   # checkpoint time; set from the opening pass

    def __post_init__(self):
        if self.entry_time is None and self.passes:
            self.entry_time = self.passes[0].ref

    @property
    def first_seen(self) -> datetime:
        """When the truck first appears on any camera (includes waiting time)."""
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
    merge_gap_by_camera: Optional[Mapping[str, float]] = None,
    openers: Optional[set[str]] = None,
    dropped: Optional[list[CameraPass]] = None,
) -> list[Event]:
    """Group camera passes into events; see the module docstring for the rule.

    chain              camera ids in the order a truck reaches them;
                       chain[0] is the head camera. Passes on cameras not
                       in the chain are ignored.
    max_lag_s          latest a later-camera pass's reference time may be
                       after its event's entry time.
    merge_gap_s        largest gap between fragments of one truck on one
                       later camera.
    head_merge_gap_s   head-camera tracks this close together are one truck
                       whose track broke (rare on 561, kept small so two
                       real trucks never merge).
    clock_tolerance_s  a pass's reference time may be this much *before*
                       its event, to absorb residual clock offset.
    merge_gap_by_camera  per-camera override of merge_gap_s. The side
                       cameras need longer: while the container side fills
                       their frame the detector sees no truck, so the cab
                       and rear fragments of one truck are ~25-35 s apart.
    openers            cameras whose unmatched passes may open an event of
                       their own (None = all). The side cameras are left
                       out: every real entry truck is also seen by 561 or
                       558, while short, barely moving fragments of
                       outbound trucks slip past their direction check.
    dropped            if given, unmatched passes from non-openers are
                       appended here so the caller can report them.
    """
    if not chain:
        raise ValueError("chain must name at least the head camera")
    head = chain[0]
    order = {cam: i for i, cam in enumerate(chain)}
    tol = timedelta(seconds=clock_tolerance_s)
    # Sort by reference time, but delay later-camera passes by the clock
    # tolerance: a head pass starting up to `tol` after a later-camera pass
    # must still open its event first, or the later pass would open a
    # spurious event of its own. Ties go to the earlier camera in the chain.
    todo = sorted((p for p in passes if p.camera in order),
                  key=lambda p: (p.ref + (tol if p.camera != head else timedelta(0)),
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
            (e for e in reversed(events) if e.entry_time <= p.ref + tol), None)
        if target is not None:
            same_cam = target.on(p.camera)
            too_late = (p.ref - target.entry_time).total_seconds() > max_lag_s
            if same_cam:
                # A continuing fragment may run past max_lag_s (a truck that
                # stops under the camera); it only has to follow closely.
                gap = (merge_gap_by_camera or {}).get(p.camera, merge_gap_s)
                too_late = (p.start - same_cam[-1].end).total_seconds() > gap
            if too_late:
                target = None
        if target is not None:
            target.passes.append(p)
        elif openers is None or p.camera in openers:
            events.append(Event([p], head_seen=False))
        elif dropped is not None:
            dropped.append(p)

    return sorted(events, key=lambda e: e.entry_time)


def camera_chain(configs: Mapping[str, CameraConfig]) -> list[str]:
    """Cameras in the order a truck reaches them: roles cab -> side -> codes -> backup."""
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
                passes.append(CameraPass(cam, tid, start, end, anchor_end=cfg.anchor == "end"))
                detections[(cam, tid)] = p.detections
    unmatched: list[CameraPass] = []
    events = assemble_events(
        passes, chain,
        merge_gap_by_camera={c: configs[c].merge_gap_s for c in chain
                             if configs[c].merge_gap_s is not None},
        openers={c for c in chain if configs[c].opens_events},
        dropped=unmatched, **assembly_kwargs)
    for p in unmatched:
        rejected.append({"camera": p.camera, "track_id": p.track_id,
                         "start": f"{p.start:%H:%M:%S}", "end": f"{p.end:%H:%M:%S}",
                         "frames": len(detections.pop((p.camera, p.track_id))),
                         "reason": "matches no event (this camera cannot open events)"})
    return ExtractedEvents(chain, events, detections, rejected)
