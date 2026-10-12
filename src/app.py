
import csv
import json
import sys
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_analyzer import analyze_video


st.set_page_config(
    page_title="AI RoadGuard",
    page_icon="🚦",
    layout="wide",
)

st.title("AI RoadGuard")
st.write(
    "AI-powered road object detection and traffic analytics "
    "using OpenCV and YOLO."
)


@st.cache_resource
def load_model():
    model_source = (
        str(MODEL_PATH)
        if MODEL_PATH.exists()
        else "yolov8n.pt"
    )
    return YOLO(model_source)


try:
    model = load_model()
except Exception as exc:
    st.error("Could not load the YOLOv8 model.")
    st.exception(exc)
    st.stop()



model = load_model()

mode = st.sidebar.radio(
    "Choose analysis mode",
    ["Image Detection", "Video Traffic Analytics"],
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05,
)

image_size = st.sidebar.select_slider(
    "Inference Image Size",
    options=[320, 480, 640, 800, 960],
    value=640,
)

line_position = st.sidebar.slider(
    "Virtual Crossing Line Position (%)",
    min_value=10,
    max_value=90,
    value=65,
    step=5,
    help="Move the crossing line vertically within the video frame.",
)


def make_csv(data):
    buffer = StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["Metric", "Category", "Value"])

    for metric, value in data.items():
        if isinstance(value, dict):
            for category, count in value.items():
                writer.writerow([metric, category, count])
        elif isinstance(value, (str, int, float, bool)):
            writer.writerow([metric, "", value])

    return buffer.getvalue()


if mode == "Image Detection":
    st.header("Road Object Detection")

    uploaded_image = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png", "webp"],
        key="road_image",
    )

    if uploaded_image is not None:
        image_bytes = np.frombuffer(
            uploaded_image.getvalue(),
            dtype=np.uint8,
        )
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Could not read this image. Try another file.")
            st.stop()

        with st.spinner("Detecting road objects..."):
            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )

        result = results[0]
        annotated_image = result.plot()
        annotated_rgb = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB,
        )

        class_counts = {}
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                label = model.names[class_id]

                class_counts[label] = (
                    class_counts.get(label, 0) + 1
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value in box.xyxy[0].tolist()
                ]

                detections.append({
                    "class": label,
                    "confidence": round(score, 4),
                    "bounding_box": [x1, y1, x2, y2],
                })

        summary = {
            "mode": "image",
            "filename": uploaded_image.name,
            "confidence_threshold": confidence,
            "image_size": image_size,
            "total_detections": len(detections),
            "class_counts": class_counts,
            "detections": detections,
        }

        st.image(
            annotated_rgb,
            caption="Annotated Road Image",
            width="stretch",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Objects Detected", len(detections))

        with col2:
            st.metric("Object Classes Found", len(class_counts))

        st.subheader("Objects by Type")

        if class_counts:
            st.bar_chart(class_counts)
        else:
            st.info("No objects detected at this confidence threshold.")

        with st.expander("Detection Details"):
            st.json(summary)

        image_output = cv2.imencode(
            ".jpg",
            annotated_image,
        )[1].tobytes()

        st.download_button(
            "Download Annotated Image",
            data=image_output,
            file_name="roadguard_detected.jpg",
            mime="image/jpeg",
        )

        st.download_button(
            "Download Detection JSON",
            data=json.dumps(summary, indent=2),
            file_name="roadguard_detection_summary.json",
            mime="application/json",
        )

        st.download_button(
            "Download Detection CSV",
            data=make_csv({
                "total_detections": len(detections),
                "class_counts": class_counts,
            }),
            file_name="roadguard_image_report.csv",
            mime="text/csv",
        )


elif mode == "Video Traffic Analytics":
    st.header("Video Traffic Analytics")

    st.write(
        "Upload a road video to detect and track vehicles, "
        "estimate congestion, count virtual-line crossings, "
        "and export a traffic report."
    )

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov", "mkv"],
        key="road_video",
    )

    if uploaded_video is not None:
        st.video(uploaded_video.getvalue())

        if st.button("Analyze Traffic Video", type="primary"):
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            input_path = OUTPUT_DIR / f"uploaded_video{suffix}"

            input_path.write_bytes(uploaded_video.getvalue())

            try:
                with st.spinner(
                    "Analyzing video. This may take a few minutes..."
                ):
                    output_video, output_json, video_summary = analyze_video(
                        str(input_path),
                        confidence=confidence,
                        image_size=image_size,
                        line_position=line_position,
                    )

                st.session_state["roadguard_video_summary"] = (
                    video_summary
                )
                st.session_state["roadguard_output_video"] = str(
                    output_video
                )
                st.session_state["roadguard_output_json"] = str(
                    output_json
                )

            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")

    video_summary = st.session_state.get(
        "roadguard_video_summary"
    )

    if video_summary:
        st.success("Video analysis completed.")

        output_video_path = Path(
            st.session_state["roadguard_output_video"]
        )
        output_json_path = Path(
            st.session_state["roadguard_output_json"]
        )

        if output_video_path.exists():
            st.subheader("Annotated Video")
            st.video(str(output_video_path))

            st.download_button(
                "Download Annotated Video",
                data=output_video_path.read_bytes(),
                file_name="roadguard_video_result.mp4",
                mime="video/mp4",
            )
        else:
            st.warning("The annotated video file was not found.")

        total_crossings = video_summary.get(
            "total_vehicle_line_crossings", 0
        )
        unique_objects = video_summary.get(
            "unique_tracked_objects",
            video_summary.get("unique_tracked_vehicles", 0),
        )
        processed_frames = video_summary.get(
            "processed_frames", 0
        )
        total_detections = video_summary.get(
            "total_detections_across_frames",
            video_summary.get(
                "total_detections",
                video_summary.get("total_vehicle_detections", 0),
            ),
        )

        st.subheader("Traffic Analytics Dashboard")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.metric("Line Crossings", total_crossings)

        with col2:
            st.metric("Unique Tracked Objects", unique_objects)

        with col3:
            st.metric("Frames Processed", processed_frames)

        with col4:
            st.metric("Total Frame Detections", total_detections)

        congestion = video_summary.get(
            "congestion_estimation", {}
        )
        st.subheader("Road-Relevant Class Counts")

        road_class_counts = video_summary.get(
            "road_relevant_class_counts", {}
        )

        if road_class_counts:
            st.bar_chart(road_class_counts)
            st.caption(
                "Counts represent detections across processed frames, "
                "not unique vehicles. People are included in this "
                "road-relevant summary."
            )
        else:
            st.info(
                "Road-relevant class counts are unavailable in this "
                "summary. Re-run video analysis to generate them."
            )
        st.subheader("Traffic Congestion Estimation")

        if congestion:
            average_vehicles = congestion.get(
                "average_vehicles_per_frame", 0
            )
            peak_vehicles = congestion.get(
                "peak_vehicles_in_one_frame", 0
            )
            congestion_level = congestion.get(
                "estimated_overall_level", "Unknown"
            )
            frames_by_level = congestion.get(
                "frames_by_level", {}
            )

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "Estimated Congestion",
                    congestion_level,
                )

            with metric2:
                st.metric(
                    "Average Vehicles per Frame",
                    f"{average_vehicles:.2f}",
                )

            with metric3:
                st.metric(
                    "Peak Vehicles in One Frame",
                    peak_vehicles,
                )

            st.caption(
                "Congestion is estimated from vehicle detections per frame. "
                "These heuristic levels are not calibrated traffic "
                "measurements and may vary with camera angle and visibility."
            )

            st.subheader("Frames by Congestion Level")

            chart_data = {
                level: frames_by_level.get(level, 0)
                for level in ["Low", "Moderate", "High"]
            }

            st.bar_chart(chart_data)

            with st.expander("Congestion Method and Thresholds"):
                st.write(
                    congestion.get(
                        "method",
                        "Method not specified.",
                    )
                )
                st.write(
                    "Vehicle classes included:",
                    ", ".join(
                        congestion.get(
                            "vehicle_classes_included", []
                        )
                    ),
                )
                st.write(
                    "Low level threshold:",
                    congestion.get(
                        "low_threshold_max_vehicles", "Unknown"
                    ),
                )
                st.write(
                    "Moderate level threshold:",
                    congestion.get(
                        "moderate_threshold_max_vehicles", "Unknown"
                    ),
                )
                st.write(congestion.get("interpretation", ""))

        else:
            st.info(
                "No congestion estimates were found in this report. "
                "Ensure src/video_analyzer.py includes congestion estimation."
            )

        directions = video_summary.get(
            "crossings_by_direction", {}
        )
        vehicle_classes = video_summary.get(
            "crossings_by_vehicle_class", {}
        )

        st.subheader("Crossings by Direction")

        if directions:
            st.bar_chart(directions)
        else:
            st.info("No directional crossing data is available.")

        st.subheader("Crossings by Vehicle Type")

        if vehicle_classes:
            st.bar_chart(vehicle_classes)
        else:
            st.info("No vehicle-class crossing data is available.")

        st.subheader("Full Analysis Summary")

        with st.expander("View Video Summary JSON"):
            st.json(video_summary)

        if output_json_path.exists():
            st.download_button(
                "Download Full JSON Report",
                data=output_json_path.read_bytes(),
                file_name="roadguard_video_summary.json",
                mime="application/json",
            )
        else:
            st.download_button(
                "Download Full JSON Report",
                data=json.dumps(video_summary, indent=2),
                file_name="roadguard_video_summary.json",
                mime="application/json",
            )

        report_data = {
            "total_line_crossings": total_crossings,
            "unique_tracked_objects": unique_objects,
            "processed_frames": processed_frames,
            "total_frame_detections": total_detections,
            "crossings_by_direction": directions,
            "crossings_by_vehicle_type": vehicle_classes,
        }

        if congestion:
            report_data["congestion_estimation"] = {
                "average_vehicles_per_frame": congestion.get(
                    "average_vehicles_per_frame", 0
                ),
                "peak_vehicles_in_one_frame": congestion.get(
                    "peak_vehicles_in_one_frame", 0
                ),
                "estimated_overall_level": congestion.get(
                    "estimated_overall_level", "Unknown"
                ),
                "frames_by_level": congestion.get(
                    "frames_by_level", {}
                ),
            }

        st.download_button(
            "Download Traffic Analytics CSV",
            data=make_csv(report_data),
            file_name="roadguard_traffic_analytics.csv",
            mime="text/csv",
        )

st.sidebar.divider()
st.sidebar.caption("AI RoadGuard | OpenCV AI Competition 2026")
