"""Tests for Malaysian plate parsing and per-truck plate fusion."""

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.plates import (
    TRAILER, VEHICLE, Fragment, PlateRead, fuse_plates, normalise_plate, plates_in_frame,
)


def frag(text, x1, y1, x2, y2, conf=0.9):
    return Fragment(text, conf, x1, y1, x2, y2)


class TestNormalise:
    def test_clean_vehicle_plates(self):
        assert normalise_plate("BQB 7404") == ("BQB7404", VEHICLE)
        assert normalise_plate("WDX7232") == ("WDX7232", VEHICLE)
        assert normalise_plate("PKN 1234 A") == ("PKN1234A", VEHICLE)

    def test_trailer_plate(self):
        assert normalise_plate("T/BF 7421") == ("T/BF7421", TRAILER)

    def test_lookalikes_fixed_by_position(self):
        assert normalise_plate("BQB 74O4") == ("BQB7404", VEHICLE)     # O in digits
        assert normalise_plate("8QB7404") == ("BQB7404", VEHICLE)      # 8 in letters

    def test_fewest_fixes_wins(self):
        assert normalise_plate("SG328") == ("SG328", VEHICLE)          # not S6328

    def test_rejects_non_plates(self):
        assert normalise_plate("INF403") is None        # I is not used on plates
        assert normalise_plate("KENDERAAN") is None
        assert normalise_plate("22G1") is None
        assert normalise_plate("7080") is None

    def test_short_words_rejected(self):
        assert normalise_plate("CUBE") is None          # would be CU8E with one fix
        assert normalise_plate("BQ8") is None           # partial read
        assert normalise_plate("G410") is None          # one letter


class TestFrame:
    def test_two_line_trailer_plate_joined(self):
        frags = [frag("T/BF", 0.30, 0.79, 0.35, 0.82), frag("7421", 0.30, 0.80, 0.35, 0.83)]
        assert [r.plate for r in plates_in_frame(frags)] == ["T/BF7421"]

    def test_side_by_side_fragments_not_stacked(self):
        frags = [frag("T/BF", 0.30, 0.80, 0.35, 0.82), frag("7421", 0.60, 0.80, 0.65, 0.82)]
        assert plates_in_frame(frags) == []

    def test_fleet_number_high_on_cab_ignored(self):
        frags = [frag("SG328", 0.40, 0.05, 0.60, 0.15), frag("BRT 3928", 0.45, 0.70, 0.55, 0.74)]
        assert [r.plate for r in plates_in_frame(frags, min_y=0.4)] == ["BRT3928"]


def pr(plate, frame, camera="561", kind=VEHICLE):
    return PlateRead(plate, kind, 0.9, camera, frame)


class TestFuse:
    def test_majority_of_frames(self):
        reads = [pr("BQB7404", i) for i in range(3)] + [pr("BQB7484", 9)]
        out = fuse_plates(reads, VEHICLE)
        assert (out.plate, out.support, out.frames) == ("BQB7404", 3, 4)

    def test_single_read_rejected(self):
        out = fuse_plates([pr("BQB7404", 0)], VEHICLE)
        assert out.plate is None and "rejected" in out.note

    def test_tie_gives_none(self):
        reads = [pr("BQB7404", 0), pr("BQB7404", 1), pr("BQB7484", 2), pr("BQB7484", 3)]
        assert fuse_plates(reads, VEHICLE).plate is None

    def test_kinds_fused_separately(self):
        reads = [pr("BQB7404", 0), pr("BQB7404", 1),
                 pr("T/BF7421", 5, "558", TRAILER), pr("T/BF7421", 6, "558", TRAILER)]
        assert fuse_plates(reads, VEHICLE).plate == "BQB7404"
        out = fuse_plates(reads, TRAILER)
        assert out.plate == "T/BF7421" and out.cameras == ("558",)

    def test_trailer_misreads_dropped_from_vehicle_vote(self):
        reads = [pr("BQB7404", 0), pr("BQB7404", 1),
                 pr("TBF7421", 2, "558"), pr("TBF7421", 3, "558"), pr("UBF7421", 4, "558"),
                 pr("T/BF7421", 5, "557", TRAILER)]
        out = fuse_plates(reads, VEHICLE)
        assert out.plate == "BQB7404" and out.frames == 2

    def test_evidence_is_most_confident_read(self):
        reads = [PlateRead("BQB7404", VEHICLE, 0.5, "561", 0), PlateRead("BQB7404", VEHICLE, 0.95, "557", 7)]
        ev = fuse_plates(reads, VEHICLE).evidence
        assert (ev.camera, ev.frame) == ("557", 7)

    def test_stacked_plate_box_spans_both_lines(self):
        frags = [frag("T/BF", 0.30, 0.79, 0.35, 0.82), frag("7421", 0.31, 0.80, 0.36, 0.83)]
        assert plates_in_frame(frags)[0].box == (0.30, 0.79, 0.36, 0.83)

    def test_one_vote_per_frame(self):
        reads = [pr("BQB7404", 0), pr("BQB7404", 0), pr("BQB7484", 1), pr("BQB7484", 2)]
        assert fuse_plates(reads, VEHICLE).plate == "BQB7484"
