"""Tests for multi-frame / cross-camera code fusion."""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest

from src.fusion import FrameRead, StatusWeightedVoteFuser, best_reading

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

    def test_single_verified_frame_is_enough(self):
        out = self.fuser.fuse([fr("verified", A), fr("failed", frame=1)])
        assert (out.code, out.status, out.support, out.frames) == (A, "verified", 1, 2)

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
        # 1 verified (3) beats 2 inferred (2).
        out = self.fuser.fuse([fr("verified", A, frame=0),
                               fr("inferred", B, frame=1), fr("inferred", B, frame=2)])
        assert out.code == A and "also read: OKMJ5632838" in out.note

    def test_many_inferred_outvote_one_verified(self):
        out = self.fuser.fuse([fr("verified", A, frame=0)] +
                              [fr("inferred", B, frame=i) for i in range(1, 5)])
        assert out.code == B and out.status == "inferred"

    def test_tie_gives_no_code(self):
        out = self.fuser.fuse([fr("verified", A, frame=0), fr("verified", B, frame=1)])
        assert out.code is None and out.note.startswith("tie")

    def test_cross_camera_support(self):
        out = self.fuser.fuse([fr("verified", A, camera="558"), fr("recovered", A, camera="559")])
        assert out.cameras == ("558", "559") and out.support == 2

    def test_size_type_voted_separately(self):
        out = self.fuser.fuse([fr("failed", frame=0, size_type="22G1"),
                               fr("failed", frame=1, size_type="22G1"),
                               fr("verified", A, frame=2, size_type="45G1")])
        assert (out.code, out.size_type, out.size_type_support) == (A, "22G1", 2)

    def test_size_type_tie_is_none(self):
        out = self.fuser.fuse([fr("failed", size_type="22G1"), fr("failed", frame=1, size_type="45G1")])
        assert out.size_type is None

    def test_no_reads(self):
        out = self.fuser.fuse([])
        assert out.code is None and out.status == "failed" and out.frames == 0

    def test_configurable_support(self):
        lenient = StatusWeightedVoteFuser(min_inferred_support=1)
        assert lenient.fuse([fr("inferred", B)]).code == B
