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