"""Tests for multi-frame / cross-camera code fusion."""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from datetime import datetime, timedelta

from src.fusion import (FrameRead, FusedRead, MultiContainerFuser, StatusWeightedVoteFuser,
                        best_reading, resolve_shared_codes)

A = "EMCU6461774"
B = "OKMJ5632838"


def fr(status, code=None, frame=0, camera="558", size_type=None, edits=0, conf=0.5, rotation=0):
    return FrameRead(camera, frame, status, code, size_type, edits, conf, rotation)


class TestBestReading:
    def test_status_beats_confidence(self):
        best = best_reading([fr("inferred", B, conf=0.99, rotation=0),
                             fr("verified", A, conf=0.30, rotation=270)])
        assert best.code == A and best.rotation == 270

    def test_fewer_edits_then_confidence(self):
        assert best_reading([fr("recovered", A, edits=2, conf=0.9),
                             fr("recovered", B, edits=1, conf=0.1)]).code == B
        assert best_reading([fr("verified", A, conf=0.4),
                             fr("verified", B, conf=0.8)]).code == B

    def test_first_wins_full_tie(self):
        assert best_reading([fr("failed", rotation=0), fr("failed", rotation=90)]).rotation == 0

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            best_reading([])


class TestVoteFuser:
    fuser = StatusWeightedVoteFuser()

    def test_single_verified_frame_is_not_enough(self):
        # 13 Nov 11:00: three single-frame codes (one verified, one recovered)
        # on a bare chassis and two prime movers without a trailer.
        out = self.fuser.fuse([fr("verified", A), fr("failed", frame=1)])
        assert out.code is None and out.note == f"rejected: {A} verified on 1 frame(s)"

    def test_two_verified_frames_give_a_code(self):
        out = self.fuser.fuse([fr("verified", A), fr("recovered", A, frame=1), fr("failed", frame=2)])
        assert (out.code, out.status, out.support, out.frames) == (A, "verified", 2, 3)

    def test_lone_inferred_code_rejected(self):
        # The OKMJ5632838 case: stray text inferred on one frame must not become a code.
        out = self.fuser.fuse([fr("inferred", B), fr("failed", frame=1)])
        assert out.code is None and out.status == "failed"
        assert "OKMJ5632838" in out.note

    def test_inferred_needs_two_agreeing_frames(self):
        out = self.fuser.fuse([fr("inferred", A, frame=0), fr("inferred", A, frame=1)])
        assert (out.code, out.status, out.support) == (A, "inferred", 2)

    def test_status_is_strongest_supporting_frame(self):
        out = self.fuser.fuse([fr("inferred", A, frame=0), fr("recovered", A, frame=1)])
        assert out.status == "recovered"

    def test_weights_favour_validated_reads(self):
        # 2 verified (6) beat 3 inferred (3).
        out = self.fuser.fuse([fr("verified", A, frame=0), fr("verified", A, frame=3),
                               fr("inferred", B, frame=1), fr("inferred", B, frame=2),
                               fr("inferred", B, frame=4)])
        assert out.code == A and "also read: OKMJ5632838" in out.note

    def test_many_inferred_outvote_one_verified(self):
        out = self.fuser.fuse([fr("verified", A, frame=0)] +
                              [fr("inferred", B, frame=i) for i in range(1, 5)])
        assert out.code == B and out.status == "inferred"

    def test_tie_gives_no_code(self):
        out = self.fuser.fuse([fr("verified", A, frame=0), fr("verified", B, frame=1),
                               fr("verified", A, frame=2), fr("verified", B, frame=3)])
        assert out.code is None and out.note.startswith("tie")

    def test_cross_camera_support(self):
        out = self.fuser.fuse([fr("verified", A, camera="558"), fr("recovered", A, camera="559")])
        assert out.cameras == ("558", "559") and out.support == 2

    def test_size_type_voted_separately(self):
        out = self.fuser.fuse([fr("failed", frame=0, size_type="22G1"),
                               fr("failed", frame=1, size_type="22G1"),
                               fr("failed", frame=4, size_type="22G1"),
                               fr("verified", A, frame=2, size_type="45G1"),
                               fr("verified", A, frame=3, size_type="45G1")])
        assert (out.code, out.size_type, out.size_type_support) == (A, "22G1", 3)

    def test_size_type_tie_is_none(self):
        out = self.fuser.fuse([fr("failed", size_type="22G1"), fr("failed", frame=1, size_type="45G1")])
        assert out.size_type is None

    def test_no_reads(self):
        out = self.fuser.fuse([])
        assert out.code is None and out.status == "failed" and out.frames == 0

    def test_configurable_support(self):
        lenient = StatusWeightedVoteFuser(min_inferred_support=1, min_support=1)
        assert lenient.fuse([fr("inferred", B)]).code == B
        assert StatusWeightedVoteFuser(min_support=1).fuse([fr("verified", A)]).code == A
        assert StatusWeightedVoteFuser(min_support=3).fuse(
            [fr("verified", A), fr("verified", A, frame=1)]).code is None

    def test_single_fuser_codes_tuple(self):
        assert self.fuser.fuse([fr("verified", A), fr("verified", A, frame=1)]).codes == (A,)
        assert self.fuser.fuse([]).codes == ()


C = "MSKU3398188"


def frames(code, n, status="verified", size_type=None, start=0, camera="557"):
    return [fr(status, code, frame=start + i, camera=camera, size_type=size_type) for i in range(n)]


class TestMultiContainerFuser:
    fuser = MultiContainerFuser()

    def test_second_20ft_container_added(self):
        # Event 6: rear code from 558, front code on many side frames.
        reads = frames(A, 30, size_type="22G1") + frames(C, 25, start=100)
        out = self.fuser.fuse(reads)
        assert out.code == A and out.codes == (A, C)
        assert "second container" in out.note

    def test_few_validated_misreads_not_added(self):
        # Event 15: a misread passed recovery on 2 frames.
        reads = frames(A, 10, size_type="22G1") + frames(C, 2, status="recovered", start=100)
        assert self.fuser.fuse(reads).codes == (A,)

    def test_inferred_reads_never_make_a_second_code(self):
        reads = frames(A, 10, size_type="22G1") + frames(C, 9, status="inferred", start=100)
        assert self.fuser.fuse(reads).codes == (A,)

    def test_40ft_container_has_no_second(self):
        reads = frames(A, 30, size_type="45G1") + frames(C, 25, start=100)
        assert self.fuser.fuse(reads).codes == (A,)

    def test_at_most_two_codes(self):
        reads = (frames(A, 30, size_type="22G1") + frames(C, 20, start=100)
                 + frames(B, 10, start=200))
        assert self.fuser.fuse(reads).codes == (A, C)

    def test_no_primary_no_codes(self):
        assert self.fuser.fuse(frames(B, 1, status="inferred")).codes == ()

    def test_second_code_read_as_40ft_is_rejected(self):
        # 13 Nov 11:00, event 10: the 45G1 container of the truck ahead was
        # read on 11 side-camera frames of a 22G1 truck.
        reads = frames(A, 30, size_type="22G1") + frames(C, 11, size_type="45G1", start=100)
        out = self.fuser.fuse(reads)
        assert out.codes == (A,)
        assert f"not a second container: {C} reads as 45G1" in out.note

    def test_second_code_read_as_20ft_is_kept(self):
        reads = frames(A, 30, size_type="22G1") + frames(C, 8, size_type="22G1", start=100)
        assert self.fuser.fuse(reads).codes == (A, C)


class TestResolveSharedCodes:
    """A container listed by trucks close in time goes to one of them."""

    T0 = datetime(2026, 11, 13, 11, 39, 2)

    def fused(self, code, *extra, score=None):
        scores = {code: 30.0}
        scores.update(score or {})
        return FusedRead(code, "verified", support=10, scores=scores, codes=(code, *extra))

    def times(self, *offsets_s):
        return {i + 1: self.T0 + timedelta(seconds=s) for i, s in enumerate(offsets_s)}

    def test_second_container_owned_by_neighbour_is_dropped(self):
        # Event 9 (TXGU...) and event 10, 15 s later, listing it as second container.
        out = {1: self.fused(C), 2: self.fused(A, C, score={C: 33.0})}
        resolve_shared_codes(out, self.times(0, 15))
        assert out[1].codes == (C,)
        assert out[2].codes == (A,)
        assert f"dropped second container {C} (main code of event 1)" in out[2].note

    def test_far_apart_trucks_keep_their_codes(self):
        # In at the gate, out an hour later: the same container on two trucks is fine.
        out = {1: self.fused(C), 2: self.fused(A, C)}
        resolve_shared_codes(out, self.times(0, 3600))
        assert out[2].codes == (A, C)

    def test_second_container_on_two_trucks_goes_to_higher_score(self):
        out = {1: self.fused(A, C, score={C: 24.0}), 2: self.fused(B, C, score={C: 9.0})}
        resolve_shared_codes(out, self.times(0, 20))
        assert out[1].codes == (A, C)
        assert out[2].codes == (B,)
        assert "more support on event 1" in out[2].note

    def test_tie_drops_from_both(self):
        out = {1: self.fused(A, C, score={C: 12.0}), 2: self.fused(B, C, score={C: 12.0})}
        resolve_shared_codes(out, self.times(0, 20))
        assert out[1].codes == (A,) and out[2].codes == (B,)

    def test_same_main_code_is_flagged_not_changed(self):
        out = {1: self.fused(C), 2: self.fused(C), 3: self.fused(A, C)}
        resolve_shared_codes(out, self.times(0, 30, 60))
        assert out[1].codes == (C,) and out[2].codes == (C,)
        assert f"{C} is also the code of event 2" in out[1].note
        assert out[3].codes == (A,)

    def test_window_counts_only_trucks_listing_the_code(self):
        # Trucks 1 and 3 both list C but are 300 s apart; truck 2 between them
        # does not list it, so it does not link them.
        out = {1: self.fused(C), 2: self.fused(B), 3: self.fused(A, C, score={C: 20.0})}
        resolve_shared_codes(out, self.times(0, 150, 300))
        assert out[3].codes == (A, C)


def test_rejected_rival_with_equal_score_still_ties():
    # 13 Nov 11:00 event 16: inferred on 2 frames (score 2) vs recovered on 1 (score 2).
    out = StatusWeightedVoteFuser().fuse([fr("inferred", B, frame=0), fr("inferred", B, frame=1),
                                          fr("recovered", A, frame=2)])
    assert out.code is None and out.note == f"tie: {B} vs {A}"


def test_weaker_rejected_rival_does_not_block():
    out = StatusWeightedVoteFuser().fuse([fr("verified", A, frame=0), fr("verified", A, frame=1),
                                          fr("verified", B, frame=2)])
    assert out.code == A
