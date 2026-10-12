
import cv2
import numpy as np
import pytest

from src.video_analyzer import (
    calculate_optical_flow,
    draw_motion_vectors,
    filter_road_relevant_class_counts,
)



def test_calculate_optical_flow_returns_expected_shapes():
    previous_gray = np.zeros((100, 100), dtype=np.uint8)
    current_gray = np.zeros((100, 100), dtype=np.uint8)

    flow, average_magnitude = calculate_optical_flow(
        previous_gray,
        current_gray,
    )

    assert flow.shape == (100, 100, 2)
    assert np.isfinite(flow).all()
    assert isinstance(average_magnitude, float)
    assert np.isfinite(average_magnitude)
    assert average_magnitude >= 0




def test_calculate_optical_flow_detects_motion():
    previous_gray = np.zeros((100, 100), dtype=np.uint8)
    current_gray = np.zeros((100, 100), dtype=np.uint8)

    cv2.rectangle(
        previous_gray,
        (20, 30),
        (40, 50),
        255,
        -1,
    )

    cv2.rectangle(
        current_gray,
        (25, 30),
        (45, 50),
        255,
        -1,
    )

    flow, average_magnitude = calculate_optical_flow(
        previous_gray,
        current_gray,
    )

    assert flow.shape == (100, 100, 2)
    assert np.isfinite(flow).all()
    assert np.isfinite(average_magnitude)
    assert average_magnitude > 0



def test_draw_motion_vectors_preserves_frame_shape():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    flow = np.zeros((100, 100, 2), dtype=np.float32)

    flow[:, :, 0] = 2.0

    result = draw_motion_vectors(frame, flow)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_draw_motion_vectors_handles_zero_flow():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    flow = np.zeros((100, 100, 2), dtype=np.float32)

    result = draw_motion_vectors(frame, flow)

    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_calculate_optical_flow_rejects_invalid_frames():
    previous_gray = np.zeros((100, 100), dtype=np.uint8)
    current_gray = np.zeros((80, 100), dtype=np.uint8)

    with pytest.raises(cv2.error):
        calculate_optical_flow(
            previous_gray,
            current_gray,
        )


def test_filter_road_relevant_class_counts():
    counts = {
        "car": 12,
        "person": 4,
        "book": 1,
        "toilet": 2,
        "bicycle": 3,
    }

    assert filter_road_relevant_class_counts(counts) == {
        "bicycle": 3,
        "car": 12,
        "person": 4,
    }
