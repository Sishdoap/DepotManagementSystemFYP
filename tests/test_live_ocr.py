"""Tests for reading events from spooled crops (src/live_ocr.py), with a fake OCR."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from src.live_ocr import (EventReads, SpooledCrop, TextBox, TextReader, fuse_event, read_crop,
                          read_event, select_frames)

CODE = "EMCU6461774"          # valid check digit


class FakeReader(TextReader):
    """Reads a code on landscape images only (rotation 0/180), plus a plate low in the box."""

    def __init__(self):
        self.calls = 0

    def read(self, image_rgb):
        self.calls += 1
        h, w = image_rgb.shape[:2]
        if w < h:
            return []
        return [TextBox("EMCU", 0.9, ((10, 10), (60, 10), (60, 20), (10, 20))),
                TextBox("646177 4", 0.9, ((70, 10), (150, 10), (150, 20), (70, 20))),
                TextBox("22G1", 0.8, ((10, 25), (50, 25), (50, 35), (10, 35))),
                # plate in the lower part of the truck box (fleet numbers sit high)
                TextBox("BQB 7404", 0.95, ((400, 250), (560, 250), (560, 280), (400, 280)))]


def crop(camera="557", frame=0, t=0.0, size=(1000, 300), track=1):
    w, h = size
    return SpooledCrop(camera, track, frame, t, (10.0, 10.0, 10.0 + w, 10.0 + h),
                       (0, 0), f"/x/{camera}_{frame}.jpg")


def img(_path=None):
    return np.zeros((330, 1100, 3), dtype=np.uint8)   # box 1000x300 + padding


ROLES = {"561": "cab", "557": "side", "558": "codes", "559": "backup", "556": "ignore"}


def test_select_frames_evenly_spaced_and_keeps_ends():
    cs = [crop(frame=i, t=i) for i in range(10)]
    sel = select_frames(cs, 4)
    assert [c.frame for c in sel] == [0, 3, 6, 9]
    assert select_frames(cs, None) == cs
    assert select_frames(cs, 20) == cs
    assert [c.frame for c in select_frames(cs, 1)] == [5]
    assert select_frames(cs, 0) == []


def test_read_crop_codes_and_plates_share_rotation_zero():
    reader = FakeReader()
    frs, prs = read_crop(reader, img(), crop(), codes=True, plates=True)
    assert reader.calls == 4                                    # one OCR call per rotation
    assert [f.rotation for f in frs] == [0, 90, 180, 270]
    assert frs[0].code == CODE and frs[0].status == "verified" and frs[0].size_type == "22G1"
    assert frs[1].code is None                                  # portrait: nothing read
    assert [p.plate for p in prs] == ["BQB7404"]


def test_read_crop_plate_only_camera_reads_once():
    reader = FakeReader()
    frs, prs = read_crop(reader, img(), crop("561"), codes=False, plates=True)
    assert reader.calls == 1 and frs == [] and len(prs) == 1


def test_read_event_roles_size_and_budget():
    crops = ([crop("557", i, i) for i in range(30)]               # side: codes + plates
             + [crop("561", i, i) for i in range(3)]              # cab: plates only
             + [crop("559", 0, 0, size=(400, 200))]               # too small
             + [crop("556", 0, 0)])                               # ignored camera
    reader = FakeReader()
    out = read_event(reader, crops, ROLES, max_frames_per_camera=5, load=img)
    assert out.frames_read == 5 + 3
    assert {cam for cam, _ in out.code_reads} == {"557"}
    assert reader.calls == 5 * 4 + 3
    code, vehicle, trailer = fuse_event(out)
    assert code.code == CODE and code.support == 5
    assert vehicle.plate == "BQB7404" and vehicle.support == 8
    assert trailer.plate is None


def test_fuse_event_empty():
    code, vehicle, trailer = fuse_event(EventReads())
    assert code.code is None and vehicle.plate is None and trailer.plate is None


def test_unreadable_crop_is_skipped():
    out = read_event(FakeReader(), [crop()], ROLES, load=lambda p: None)
    assert out.frames_read == 0 and out.code_reads == {}
