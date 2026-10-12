import json

import pytest

from src.analyze_hazards import calculate_iou, main


def test_identical_boxes_have_iou_one():
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes_have_iou_zero():
    assert calculate_iou(
        [0, 0, 10, 10],
        [20, 20, 30, 30],
    ) == 0.0


def test_partially_overlapping_boxes():
    result = calculate_iou(
        [0, 0, 10, 10],
        [5, 5, 15, 15],
    )
    assert result == pytest.approx(1 / 7)


def test_degenerate_boxes_have_iou_zero():
    assert calculate_iou(
        [0, 0, 0, 10],
        [0, 0, 10, 10],
    ) == 0.0


def test_hazard_report_flags_vehicle_count_and_unexpected_class(
    tmp_path, monkeypatch
):
    import src.analyze_hazards as hazard_module

    detections = []

    for index in range(25):
        x = index * 20
        detections.append({
            "label": "car",
            "confidence": 0.9,
            "bounding_box": [x, 0, x + 10, 10],
        })

    detections.append({
        "label": "clock",
        "confidence": 0.8,
        "bounding_box": [600, 0, 610, 10],
    })

    input_file = tmp_path / "detection_summary.json"
    output_file = tmp_path / "hazard_report.json"

    input_file.write_text(
        json.dumps({
            "image": "test-road.png",
            "detections": detections,
        }),
        encoding="utf-8",
    )

    monkeypatch.setattr(hazard_module, "INPUT_JSON", input_file)
    monkeypatch.setattr(hazard_module, "OUTPUT_JSON", output_file)

    main()

    report = json.loads(output_file.read_text(encoding="utf-8"))

    assert report["total_detections"] == 26
    assert report["detected_vehicle_count"] == 25
    assert report["total_alerts"] == 2
    assert report["overlap_issue_count"] == 0

    alert_types = {alert["type"] for alert in report["alerts"]}
    assert "high_detected_vehicle_count" in alert_types
    assert "unexpected_object_class" in alert_types
