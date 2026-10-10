import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_CONFIDENCE = 0.25
IMAGE_SIZE = 960

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
HIGH_VEHICLE_COUNT_THRESHOLD = 25
OVERLAP_THRESHOLD = 0.80


st.set_page_config(
    page_title="AI RoadGuard",
    page_icon="🚦",
    layout="wide",
)

st.title("AI RoadGuard")
st.subheader("Computer Vision for Road Scene Analysis")

st.write(
    "Upload a road image to detect objects and generate "
    "preliminary review alerts using YOLO and OpenCV."
)

st.warning(
    "Prototype only. Alerts require human verification. "
    "This system does not confirm accidents or predict collisions."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


def calculate_iou(box_a, box_b):
    """Calculate intersection over union for two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box_a[2] - box_a[0])
        * max(0.0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0.0, box_b[2] - box_b[0])
        * max(0.0, box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def analyze_hazards(detections):
    """Generate explainable, rule-based review alerts."""
    alerts = []

    class_counts = Counter(
        detection["label"] for detection in detections
    )

    vehicle_count = sum(
        class_counts[label]
        for label in VEHICLE_CLASSES
    )

    if vehicle_count >= HIGH_VEHICLE_COUNT_THRESHOLD:
        alerts.append({
            "type": "high_detected_vehicle_count",
            "severity": "review",
            "message": (
                f"{vehicle_count} vehicles were detected. "
                "Review the scene for possible congestion."
            ),
            "evidence": {
                "vehicle_count": vehicle_count,
                "threshold": HIGH_VEHICLE_COUNT_THRESHOLD,
            },
        })

    for detection in detections:
        if detection["label"] == "clock":
            alerts.append({
                "type": "unexpected_object_class",
                "severity": "review",
                "message": (
                    "A clock was detected in the road image. "
                    "Check whether this is a false positive."
                ),
                "evidence": detection,
            })

    overlap_issue_count = 0

    for first, second in combinations(detections, 2):
        iou = calculate_iou(
            first["bounding_box"],
            second["bounding_box"],
        )

        if iou >= OVERLAP_THRESHOLD:
            overlap_issue_count += 1

            if first["label"] == second["label"]:
                issue_type = "possible_duplicate_detection"
            else:
                issue_type = "overlapping_different_classes"

            alerts.append({
                "type": issue_type,
                "severity": "review",
                "message": (
                    f"Inspect overlapping detections: "
                    f"{first['label']} and {second['label']} "
                    f"(IoU: {iou:.2f})."
                ),
                "evidence": {
                    "labels": [
                        first["label"],
                        second["label"],
                    ],
                    "iou": round(iou, 4),
                    "first_box": first["bounding_box"],
                    "second_box": second["bounding_box"],
                },
            })

    report = {
        "analysis_type": "rule_based_detection_review",
        "total_detections": len(detections),
        "class_counts": dict(sorted(class_counts.items())),
        "detected_vehicle_count": vehicle_count,
        "overlap_issue_count": overlap_issue_count,
        "total_alerts": len(alerts),
        "alerts": alerts,
        "limitations": [
            "A high vehicle count does not prove congestion.",
            "Overlapping boxes may be valid objects or model errors.",
            "Alerts require human verification.",
            "The system does not confirm collisions or predict accidents.",
        ],
    }

    return report


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
    help=(
        "Higher values filter out more low-confidence detections. "
        "This can also remove valid objects."
    ),
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()

    height, width = image.shape[:2]

    st.caption(f"Image dimensions: {width} × {height}")

    original_column, result_column = st.columns(2)

    with original_column:
        st.markdown("### Original image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    if st.button("Analyze road image", type="primary"):
        try:
            with st.spinner("Running YOLO object detection..."):
                model = load_model()

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=IMAGE_SIZE,
                    verbose=False,
                )

                result = results[0]
                annotated_image = result.plot()

                detections = []

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())

                    detections.append({
                        "label": model.names[class_id],
                        "confidence": round(score, 4),
                        "bounding_box": [
                            round(float(value), 2)
                            for value in box.xyxy[0].tolist()
                        ],
                    })

                class_counts = Counter(
                    detection["label"]
                    for detection in detections
                )

                hazard_report = analyze_hazards(detections)

                detection_summary = {
                    "image": uploaded_file.name,
                    "image_width": width,
                    "image_height": height,
                    "model": MODEL_PATH.name,
                    "confidence_threshold": confidence,
                    "image_size": IMAGE_SIZE,
                    "total_detections": len(detections),
                    "class_counts": dict(
                        sorted(class_counts.items())
                    ),
                    "detections": detections,
                }

                hazard_report["image"] = uploaded_file.name
                hazard_report["confidence_threshold"] = confidence

                OUTPUT_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                image_output_path = (
                    OUTPUT_DIR / "roadguard_app_result.jpg"
                )

                image_saved = cv2.imwrite(
                    str(image_output_path),
                    annotated_image,
                )

                if not image_saved:
                    raise RuntimeError(
                        "Could not save the annotated image."
                    )

                (OUTPUT_DIR / "app_detection_summary.json").write_text(
                    json.dumps(detection_summary, indent=2),
                    encoding="utf-8",
                )

                (OUTPUT_DIR / "app_hazard_report.json").write_text(
                    json.dumps(hazard_report, indent=2),
                    encoding="utf-8",
                )

                st.session_state["roadguard_result"] = {
                    "image": annotated_image,
                    "summary": detection_summary,
                    "report": hazard_report,
                    "source_name": uploaded_file.name,
                }

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")


if "roadguard_result" in st.session_state:
    result_data = st.session_state["roadguard_result"]
    summary = result_data["summary"]
    report = result_data["report"]

    st.divider()
    st.markdown("## Analysis results")

    with result_column:
        st.markdown("### Detected objects")
        st.image(
            cv2.cvtColor(
                result_data["image"],
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Objects detected",
        summary["total_detections"],
    )

    metric2.metric(
        "Vehicles detected",
        report["detected_vehicle_count"],
    )

    metric3.metric(
        "Overlap issues",
        report["overlap_issue_count"],
    )

    metric4.metric(
        "Review alerts",
        report["total_alerts"],
    )

    st.markdown("### Object counts")

    if summary["class_counts"]:
        st.dataframe(
            [
                {"Object": label, "Count": count}
                for label, count in summary["class_counts"].items()
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No objects were detected at the selected confidence threshold."
        )

    st.markdown("### Review alerts")

    if report["alerts"]:
        for index, alert in enumerate(
            report["alerts"],
            start=1,
        ):
            with st.expander(
                f"{index}. {alert['type']}",
                expanded=index == 1,
            ):
                st.warning(alert["message"])
                st.caption(
                    f"Severity: {alert['severity']}"
                )

                if "evidence" in alert:
                    st.json(alert["evidence"])
    else:
        st.success(
            "No alerts were generated by the current rules."
        )

    st.caption(
        "These rules identify items for review, not confirmed hazards. "
        "A lack of alerts does not establish that a road is safe."
    )

    st.markdown("### Download reports")

    download_column1, download_column2 = st.columns(2)

    with download_column1:
        st.download_button(
            "Download detection summary",
            data=json.dumps(summary, indent=2),
            file_name="detection_summary.json",
            mime="application/json",
            use_container_width=True,
        )

    with download_column2:
        st.download_button(
            "Download hazard report",
            data=json.dumps(report, indent=2),
            file_name="hazard_report.json",
            mime="application/json",
            use_container_width=True,
        )

    st.download_button(
        "Download annotated image",
        data=cv2.imencode(
            ".jpg",
            result_data["image"],
        )[1].tobytes(),
        file_name="roadguard_result.jpg",
        mime="image/jpeg",
    )