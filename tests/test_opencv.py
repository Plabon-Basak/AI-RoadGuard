import cv2
import numpy as np
import pytest

from src.video_analyzer import (
    calculate_optical_flow,
    draw_motion_vectors,
)


def test_optical_flow_returns_valid_flow_and_score():
    previous = np.zeros((120, 160), dtype=np.uint8)
    current = previous.copy()

    cv2.rectangle(previous, (30, 30), (70, 70), 255, -1)
    cv2.rectangle(current, (35, 30), (75, 70), 255, -1)

    flow, motion_score = calculate_optical_flow(previous, current)

    assert flow.shape == (120, 160, 2)
    assert np.isfinite(flow).all()
    assert np.isfinite(motion_score)
    assert motion_score >= 0


def test_optical_flow_detects_motion_between_frames():
    previous = np.zeros((120, 160), dtype=np.uint8)
    current = previous.copy()

    cv2.rectangle(previous, (30, 30), (70, 70), 255, -1)
    cv2.rectangle(current, (40, 30), (80, 70), 255, -1)

    _, motion_score = calculate_optical_flow(previous, current)

    assert motion_score > 0


def test_draw_motion_vectors_accepts_analysis_dimensions():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    flow = np.zeros((60, 80, 2), dtype=np.float32)

    flow[:, :, 0] = 2.0

    result = draw_motion_vectors(
        frame,
        flow,
        80,
        60,
    )

    assert result is frame
    assert result.shape == (240, 320, 3)
    assert np.any(result != 0)


def test_draw_motion_vectors_handles_zero_motion():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    flow = np.zeros((60, 80, 2), dtype=np.float32)

    result = draw_motion_vectors(
        frame,
        flow,
        80,
        60,
    )

    assert result.shape == (120, 160, 3)
    assert np.count_nonzero(result) == 0
