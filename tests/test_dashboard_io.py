"""Tests for the dashboard's frame-sharing layer."""

import numpy as np
import pytest

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dashboard_io import clear_frames, read_gate_frame, write_gate_frame


@pytest.fixture
def tmp_frames(tmp_path):
    return tmp_path / "frames"


class TestFrameSharing:
    def test_write_then_read(self, tmp_frames):
        img = np.zeros((50, 100, 3), dtype=np.uint8)
        img[:, :, 1] = 255   # green
        write_gate_frame("A", img, frame_dir=tmp_frames)
        data = read_gate_frame("A", frame_dir=tmp_frames)
        assert data is not None
        # PNG magic bytes.
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_read_missing_returns_none(self, tmp_frames):
        assert read_gate_frame("Z", frame_dir=tmp_frames) is None

    def test_overwrites_previous_frame(self, tmp_frames):
        img1 = np.full((20, 40, 3), 100, dtype=np.uint8)
        img2 = np.full((20, 40, 3), 200, dtype=np.uint8)
        write_gate_frame("A", img1, frame_dir=tmp_frames)
        data1 = read_gate_frame("A", frame_dir=tmp_frames)
        write_gate_frame("A", img2, frame_dir=tmp_frames)
        data2 = read_gate_frame("A", frame_dir=tmp_frames)
        assert data1 != data2

    def test_clear_frames(self, tmp_frames):
        img = np.zeros((20, 40, 3), dtype=np.uint8)
        write_gate_frame("A", img, frame_dir=tmp_frames)
        write_gate_frame("B", img, frame_dir=tmp_frames)
        clear_frames(tmp_frames)
        assert read_gate_frame("A", frame_dir=tmp_frames) is None
        assert read_gate_frame("B", frame_dir=tmp_frames) is None

    def test_annotation_doesnt_crash(self, tmp_frames):
        img = np.zeros((60, 200, 3), dtype=np.uint8)
        write_gate_frame("A", img, annotation="CSQU3054383 valid=True", frame_dir=tmp_frames)
        data = read_gate_frame("A", frame_dir=tmp_frames)
        assert data is not None

    def test_bounding_box_drawn(self, tmp_frames):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        write_gate_frame(
            "A", img,
            annotation="GLDU4071220",
            bounding_box=(20, 30, 80, 40),
            frame_dir=tmp_frames,
        )
        data = read_gate_frame("A", frame_dir=tmp_frames)
        assert data is not None
        # Read it back to confirm the green rectangle exists at the expected place.
        import io
        from PIL import Image as PILImage
        img_back = np.array(PILImage.open(io.BytesIO(data)))
        # At least some pixels along the box edge should be green (0, 255, 0).
        edge_pixels = img_back[30, 20:100]  # top edge
        assert (edge_pixels[:, 1] > 200).any(), "expected green pixels along bbox top edge"

class TestLiveFiles:
    """Files the live pipeline writes for the depot dashboard."""

    def test_camera_preview_round_trip(self, tmp_path):
        from src.dashboard_io import read_camera_preview, write_camera_preview
        img = np.zeros((1440, 5120, 3), np.uint8)
        write_camera_preview("557", img, [(100, 100, 900, 600, "#3")], live_dir=tmp_path,
                             label="11:06:02")
        data, mtime = read_camera_preview("557", live_dir=tmp_path)
        assert data[:2] == b"\xff\xd8" and mtime is not None
        from PIL import Image
        import io
        assert Image.open(io.BytesIO(data)).size == (1280, 360)
        assert read_camera_preview("999", live_dir=tmp_path) == (None, None)
        assert not list(tmp_path.glob("*.tmp"))

    def test_thumbnail_and_status(self, tmp_path):
        from src.dashboard_io import (read_status, thumbnail_path, write_pass_thumbnail,
                                      write_status)
        p = write_pass_thumbnail("uid1", "561", np.zeros((900, 1800, 3), np.uint8), live_dir=tmp_path)
        assert p == thumbnail_path("uid1", "561", live_dir=tmp_path) and p.exists()
        assert read_status(live_dir=tmp_path) is None
        write_status({"mode": "replay", "lag_s": 3.0}, live_dir=tmp_path)
        assert read_status(live_dir=tmp_path) == {"mode": "replay", "lag_s": 3.0}

    def test_mask_plate(self):
        from src.dashboard_io import mask_plate
        assert mask_plate("BQB7404") == "BQB•••4"
        assert mask_plate("T/BF7421") == "T/BF•••1"
        assert mask_plate("WDX7232A") == "WDX••••A"
        assert mask_plate(None) == "—"
