import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
HIGH_VEHICLE_COUNT_THRESHOLD = 25


st.set_page_config(
    page_title="AI RoadGuard",
    page_icon="🚦",
    layout="wide",
)

st.title("AI RoadGuard")
st.subheader("Computer Vision for Road Scene Analysis")

st.write(
    "Upload a road image to detect objects and generate "
    "preliminary review alerts."
)

st.warning(
    "Prototype only. Alerts require human verification and "
    "are not confirmed traffic hazards."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {MODEL_PATH}"
        )

    return YOLO(str(MODEL_PATH))


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

confidence = st.slider(
    "Detection confidence threshold",
    min_value=0.10,
    max_value=0.90,
    value=CONFIDENCE_THRESHOLD,
    step=0.05,
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8,
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    height, width = image.shape[:2]

    st.caption(f"Image dimensions: {width} × {height}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Original image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    if st.button("Analyze road image", type="primary"):
        try:
            with st.spinner("Loading model and detecting objects..."):
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
                class_counts = Counter()

                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    label = model.names[class_id]

                    coordinates = [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ]

                    detections.append({
                        "label": label,
                        "confidence": round(score, 4),
                        "bounding_box": coordinates,
                    })

                    class_counts[label] += 1

                vehicle_count = sum(
                    count
                    for label, count in class_counts.items()
                    if label in VEHICLE_CLASSES
                )

                alerts = []

                if vehicle_count >= HIGH_VEHICLE_COUNT_THRESHOLD:
                    alerts.append({
                        "type": "high_detected_vehicle_count",
                        "severity": "review",
                        "message": (
                            f"{vehicle_count} vehicles were detected. "
                            "Review the image for possible congestion."
                        ),
                    })

                for detection in detections:
                    if detection["label"] == "clock":
                        alerts.append({
                            "type": "unexpected_object_class",
                            "severity": "review",
                            "message": (
                                "A clock was detected. "
                                "Check whether this is a false positive."
                            ),
                            "evidence": detection,
                        })

                detection_summary = {
                    "image": uploaded_file.name,
                    "image_width": width,
                    "image_height": height,
                    "model": MODEL_PATH.name,
                    "confidence_threshold": confidence,
                    "total_detections": len(detections),
                    "class_counts": dict(
                        sorted(class_counts.items())
                    ),
                    "detections": detections,
                }

                hazard_report = {
                    "image": uploaded_file.name,
                    "analysis_type": "prototype_rule_based_review",
                    "total_detections": len(detections),
                    "detected_vehicle_count": vehicle_count,
                    "total_alerts": len(alerts),
                    "alerts": alerts,
                    "limitations": [
                        "Vehicle count alone does not confirm congestion.",
                        "Alerts require human verification.",
                        "This prototype does not predict collisions.",
                    ],
                }

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(
                    str(OUTPUT_DIR / "roadguard_app_result.jpg"),
                    annotated_image,
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
                }

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

if "roadguard_result" in st.session_state:
    result_data = st.session_state["roadguard_result"]
    summary = result_data["summary"]
    report = result_data["report"]

    st.divider()
    st.markdown("## Analysis results")

    with right:
        st.markdown("### Detected objects")
        st.image(
            cv2.cvtColor(
                result_data["image"],
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Objects detected",
        summary["total_detections"],
    )
    metric2.metric(
        "Vehicles detected",
        report["detected_vehicle_count"],
    )
    metric3.metric(
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
        st.info("No objects were detected at this confidence threshold.")

    st.markdown("### Review alerts")

    if report["alerts"]:
        for alert in report["alerts"]:
            st.warning(
                f"**{alert['type']}**: {alert['message']}"
            )
    else:
        st.success("No alerts were generated by the current rules.")

    st.caption(
        "No alert does not mean the road scene is safe. "
        "The current rules cover only a limited set of conditions."
    )

    st.download_button(
        "Download detection summary (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="detection_summary.json",
        mime="application/json",
    )

    st.download_button(
        "Download review report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="hazard_report.json",
        mime="application/json",
    )