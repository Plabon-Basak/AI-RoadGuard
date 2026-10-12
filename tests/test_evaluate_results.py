
import json

import pytest

from src.evaluate_results import build_evaluation, load_report


@pytest.fixture
def sample_report():
    return {
        "video": "sample.mp4",
        "model": "yolov8n.pt",
        "tracker": "ByteTrack",
        "confidence_threshold": 0.25,
        "image_size": 640,
        "video_fps": 30.0,
        "input_frame_count": 100,
        "processed_frames": 98,
        "frame_width": 1280,
        "frame_height": 720,
        "total_detections_across_frames": 250,
        "unique_tracked_objects": 12,
        "unique_tracked_vehicles": 10,
        "total_vehicle_line_crossings": 4,
        "crossings_by_direction": {"top_to_bottom": 4},
        "crossings_by_vehicle_class": {"car": 3, "bus": 1},
        "class_counts_across_frames": {"car": 200, "bus": 30, "person": 20},
        "optical_flow_analysis": {
            "enabled": True,
            "method": "Farneback dense optical flow",
            "analysis_width": 480,
            "analysis_height": 270,
            "frames_with_motion_measurement": 97,
            "average_motion_magnitude": 1.2,
            "peak_motion_magnitude": 2.8,
            "unit": "pixels per frame at analysis resolution",
            "interpretation": "Apparent pixel motion, not vehicle speed.",
        },
        "congestion_estimation": {
            "method": "Heuristic based on detected vehicles per frame",
            "vehicle_classes_included": ["car", "bus", "motorcycle", "truck"],
            "low_threshold_max_vehicles": 5,
            "moderate_threshold_max_vehicles": 12,
            "average_vehicles_per_frame": 8.5,
            "peak_vehicles_in_one_frame": 15,
            "estimated_overall_level": "Moderate",
            "frames_by_level": {"Low": 20, "Moderate": 70, "High": 8},
            "interpretation": "Prototype thresholds.",
        },
        "limitations": [
            "Tracking ID changes can affect counts.",
            "Optical flow is not calibrated speed.",
        ],
    }


def test_load_report_reads_valid_json(tmp_path, sample_report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(sample_report), encoding="utf-8")

    result = load_report(path)

    assert result["processed_frames"] == 98
    assert result["tracker"] == "ByteTrack"


def test_load_report_rejects_missing_required_field(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"video": "sample.mp4"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required report fields"):
        load_report(path)


def test_build_evaluation_includes_recorded_metrics(sample_report):
    result = build_evaluation(sample_report)

    assert "# AI RoadGuard Evaluation Report" in result
    assert "| Detections across frames | 250 |" in result
    assert "| Unique tracked vehicles | 10 |" in result
    assert "| Virtual-line crossings | 4 |" in result
    assert "Farneback dense optical flow" in result
    assert "Estimated overall level" in result


def test_build_evaluation_discloses_limitations(sample_report):
    result = build_evaluation(sample_report)

    assert "Precision and recall require ground-truth object annotations." in result
    assert "Tracking ID changes can affect counts." in result
    assert "not a direct measurement of vehicle speed" in result
