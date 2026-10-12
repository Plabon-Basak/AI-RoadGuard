import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT / "outputs" / "detection_summary.json"
OUTPUT_JSON = ROOT / "outputs" / "hazard_report.json"

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
HIGH_VEHICLE_COUNT_THRESHOLD = 25
OVERLAP_THRESHOLD = 0.80


def calculate_iou(box_a, box_b):
    """Calculate intersection over union for two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_JSON}. Run python src\\detect_objects.py first."
        )

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    detections = data.get("detections", [])
    alerts = []

    class_counts = Counter(
        item["label"] for item in detections
    )

    vehicle_count = sum(
        class_counts[label] for label in VEHICLE_CLASSES
    )

    # Rule 1: high vehicle count means the image needs review,
    # not that congestion has been confirmed.
    if vehicle_count >= HIGH_VEHICLE_COUNT_THRESHOLD:
        alerts.append({
            "type": "high_detected_vehicle_count",
            "severity": "review",
            "message": (
                f"{vehicle_count} vehicles detected. "
                "Manually inspect the scene for possible congestion."
            ),
            "evidence": {
                "vehicle_count": vehicle_count,
                "threshold": HIGH_VEHICLE_COUNT_THRESHOLD,
            },
        })

    # Rule 2: unexpected object class may be a false positive.
    for item in detections:
        if item["label"] == "clock":
            alerts.append({
                "type": "unexpected_object_class",
                "severity": "review",
                "message": (
                    "A clock was detected in this road image. "
                    "Verify whether this is a false positive."
                ),
                "evidence": item,
            })

    # Rule 3: find highly overlapping detection boxes.
    overlap_issues = []

    for first, second in combinations(detections, 2):
        iou = calculate_iou(
            first["bounding_box"],
            second["bounding_box"],
        )

        if iou >= OVERLAP_THRESHOLD:
            different_classes = first["label"] != second["label"]

            overlap_issues.append({
                "type": (
                    "overlapping_different_classes"
                    if different_classes
                    else "possible_duplicate_detection"
                ),
                "labels": [first["label"], second["label"]],
                "iou": round(iou, 4),
                "first_confidence": first["confidence"],
                "second_confidence": second["confidence"],
                "first_box": first["bounding_box"],
                "second_box": second["bounding_box"],
            })

    for issue in overlap_issues:
        alerts.append({
            "type": issue["type"],
            "severity": "review",
            "message": (
                f"Inspect overlapping detections: "
                f"{issue['labels'][0]} and {issue['labels'][1]}."
            ),
            "evidence": issue,
        })

    report = {
        "image": data.get("image"),
        "analysis_type": "rule_based_detection_review",
        "total_detections": len(detections),
        "class_counts": dict(sorted(class_counts.items())),
        "detected_vehicle_count": vehicle_count,
        "overlap_issue_count": len(overlap_issues),
        "total_alerts": len(alerts),
        "alerts": alerts,
        "limitations": [
            "A high vehicle count does not prove congestion.",
            "Overlapping boxes do not always indicate model errors.",
            "Alerts require human verification.",
            "The system does not confirm collisions or predict accidents.",
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("AI RoadGuard - Detection Review")
    print(f"Image: {report['image']}")
    print(f"Total detections: {report['total_detections']}")
    print(f"Vehicles detected: {vehicle_count}")
    print(f"Overlapping-box issues: {len(overlap_issues)}")
    print(f"Total review alerts: {len(alerts)}")

    for index, alert in enumerate(alerts, start=1):
        print(f"{index}. {alert['message']}")

    print(f"\nReport saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()