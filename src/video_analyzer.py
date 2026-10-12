
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

ROAD_RELEVANT_CLASSES = {
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "person",
}


def filter_road_relevant_class_counts(class_counts):
    return {
        label: count
        for label, count in sorted(class_counts.items())
        if label in ROAD_RELEVANT_CLASSES
    }


LOW_CONGESTION_MAX = 5
MODERATE_CONGESTION_MAX = 12

FLOW_ANALYSIS_WIDTH = 480
FLOW_VECTOR_SPACING = 32
FLOW_VECTOR_SCALE = 2.0


def estimate_congestion(vehicle_count):
    if vehicle_count <= LOW_CONGESTION_MAX:
        return "Low"
    if vehicle_count <= MODERATE_CONGESTION_MAX:
        return "Moderate"
    return "High"


def calculate_optical_flow(previous_gray, current_gray):
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    magnitude, _ = cv2.cartToPolar(
        flow[:, :, 0],
        flow[:, :, 1],
    )

    motion_score = float(cv2.mean(magnitude)[0])

    return flow, motion_score


def draw_motion_vectors(
    frame,
    flow,
    flow_width=None,
    flow_height=None,
):
    frame_height, frame_width = frame.shape[:2]
    actual_flow_height, actual_flow_width = flow.shape[:2]

    if flow_width is None:
        flow_width = actual_flow_width

    if flow_height is None:
        flow_height = actual_flow_height

    scale_x = frame_width / flow_width
    scale_y = frame_height / flow_height

    step = FLOW_VECTOR_SPACING

    for y in range(16, flow_height, step):
        for x in range(16, flow_width, step):
            fx, fy = flow[y, x]
            magnitude = (fx * fx + fy * fy) ** 0.5

            if magnitude < 0.25:
                continue

            start_x = int(x * scale_x)
            start_y = int(y * scale_y)

            end_x = int(
                (x + FLOW_VECTOR_SCALE * 4 * fx) * scale_x
            )
            end_y = int(
                (y + FLOW_VECTOR_SCALE * 4 * fy) * scale_y
            )

            start_x = max(0, min(frame_width - 1, start_x))
            start_y = max(0, min(frame_height - 1, start_y))
            end_x = max(0, min(frame_width - 1, end_x))
            end_y = max(0, min(frame_height - 1, end_y))

            start = (start_x, start_y)
            end = (end_x, end_y)

            # Dark outline keeps arrows visible on bright backgrounds.
            cv2.arrowedLine(
                frame,
                start,
                end,
                (0, 0, 0),
                7,
                cv2.LINE_AA,
                tipLength=0.4,
            )

            # Bright magenta indicates apparent motion direction.
            cv2.arrowedLine(
                frame,
                start,
                end,
                (255, 0, 255),
                3,
                cv2.LINE_AA,
                tipLength=0.4,
            )

    return frame


def draw_video_overlay(
    frame,
    line_y,
    crossing_count,
    vehicle_count,
    congestion_level,
    motion_label,
):
    height, width = frame.shape[:2]

    # Scale overlay elements for different video resolutions.
    scale = min(
        2.5,
        max(0.8, min(width, height) / 900.0),
    )

    thickness = max(2, int(2 * scale))
    line_thickness = max(3, int(3 * scale))

    # Draw a dark outline underneath the bright crossing line.
    cv2.line(
        frame,
        (0, line_y),
        (width - 1, line_y),
        (0, 0, 0),
        line_thickness + max(3, int(3 * scale)),
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        (0, line_y),
        (width - 1, line_y),
        (0, 255, 255),
        line_thickness,
        cv2.LINE_AA,
    )

    # Place a semi-transparent dark panel behind the status text.
    panel_width = min(width - 20, int(700 * scale))
    panel_height = min(height - 20, int(135 * scale))

    if panel_width > 0 and panel_height > 0:
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (15, 15, 15),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.82,
            frame,
            0.18,
            0,
            frame,
        )

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_x = 20
    first_y = int(40 * scale)
    second_y = int(77 * scale)
    third_y = int(114 * scale)

    font_scale = 0.8 * scale

    cv2.putText(
        frame,
        f"Virtual Line | Crossings: {crossing_count}",
        (text_x, first_y),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count} | Congestion: {congestion_level}",
        (text_x, second_y),
        font,
        font_scale * 0.9,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        motion_label,
        (text_x, third_y),
        font,
        font_scale * 0.8,
        (255, 0, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


def analyze_video(
    video_path,
    confidence=0.25,
    image_size=640,
    line_position=65,
    enable_optical_flow=True,
):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid frame dimensions.")

    line_position = max(10, min(90, int(line_position)))
    line_y = int(height * line_position / 100)

    output_video = OUTPUT_DIR / "roadguard_video_result.mp4"
    output_json = OUTPUT_DIR / "video_detection_summary.json"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output video.")

    model_source = (
    str(MODEL_PATH)
    if MODEL_PATH.exists()
    else "yolov8n.pt"
    )
    model = YOLO(model_source)

    class_counts = Counter()
    track_classes = {}
    previous_sides = {}
    crossed_ids = set()

    direction_counts = Counter()
    crossing_class_counts = Counter()
    frame_summaries = []

    total_detections = 0
    processed_frames = 0

    total_frame_vehicle_count = 0
    peak_frame_vehicle_count = 0
    congestion_level_counts = Counter()

    previous_flow_gray = None
    motion_scores = []

    flow_width = min(width, FLOW_ANALYSIS_WIDTH)
    flow_height = max(
        1,
        int(height * flow_width / width),
    )

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            flow_gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            if flow_width != width or flow_height != height:
                flow_gray = cv2.resize(
                    flow_gray,
                    (flow_width, flow_height),
                    interpolation=cv2.INTER_AREA,
                )

            flow = None
            motion_score = None

            if (
                enable_optical_flow
                and previous_flow_gray is not None
            ):
                flow, motion_score = calculate_optical_flow(
                    previous_flow_gray,
                    flow_gray,
                )

                motion_scores.append(motion_score)

            previous_flow_gray = flow_gray

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )[0]

            annotated_frame = result.plot()

            if flow is not None:
                draw_motion_vectors(
                    annotated_frame,
                    flow,
                    flow_width,
                    flow_height,
                )

            frame_detections = []
            frame_crossings = 0
            frame_vehicle_count = 0

            if result.boxes is not None:
                boxes = result.boxes

                tracked_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for box, track_id in zip(boxes, tracked_ids):
                    class_id = int(box.cls[0].item())
                    label = model.names[class_id]
                    score = float(box.conf[0].item())

                    class_counts[label] += 1
                    total_detections += 1

                    if label in VEHICLE_CLASSES:
                        frame_vehicle_count += 1

                    coordinates = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = coordinates

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    crossed = False
                    direction = None

                    if track_id is not None:
                        track_classes[track_id] = label

                        if label in VEHICLE_CLASSES:
                            current_side = (
                                "above"
                                if center_y < line_y
                                else "below"
                            )

                            previous_side = previous_sides.get(
                                track_id
                            )

                            if (
                                previous_side is not None
                                and previous_side != current_side
                                and track_id not in crossed_ids
                            ):
                                crossed = True
                                crossed_ids.add(track_id)
                                frame_crossings += 1

                                if (
                                    previous_side == "above"
                                    and current_side == "below"
                                ):
                                    direction = "top_to_bottom"
                                else:
                                    direction = "bottom_to_top"

                                direction_counts[direction] += 1
                                crossing_class_counts[label] += 1

                            previous_sides[track_id] = current_side

                    frame_detections.append({
                        "track_id": track_id,
                        "label": label,
                        "confidence": round(score, 4),
                        "center": [center_x, center_y],
                        "crossed_line_this_frame": crossed,
                        "crossing_direction": direction,
                        "bounding_box": [
                            round(float(value), 2)
                            for value in coordinates
                        ],
                    })

            processed_frames += 1

            congestion_level = estimate_congestion(
                frame_vehicle_count
            )

            total_frame_vehicle_count += frame_vehicle_count

            peak_frame_vehicle_count = max(
                peak_frame_vehicle_count,
                frame_vehicle_count,
            )

            congestion_level_counts[congestion_level] += 1

            if not enable_optical_flow:
                motion_label = "Optical Flow: Disabled"
            elif motion_score is None:
                motion_label = "Optical Flow: Warming up"
            else:
                motion_label = (
                    f"Optical Flow Motion: "
                    f"{motion_score:.2f} px/frame"
                )

            draw_video_overlay(
                annotated_frame,
                line_y,
                len(crossed_ids),
                frame_vehicle_count,
                congestion_level,
                motion_label,
            )

            writer.write(annotated_frame)

            frame_summaries.append({
                "frame": processed_frames,
                "timestamp_seconds": round(
                    (processed_frames - 1) / fps,
                    3,
                ),
                "detection_count": len(frame_detections),
                "vehicle_count": frame_vehicle_count,
                "congestion_level": congestion_level,
                "line_crossings_this_frame": frame_crossings,
                "optical_flow_motion_magnitude": (
                    round(motion_score, 4)
                    if motion_score is not None
                    else None
                ),
                "detections": frame_detections,
            })

    finally:
        capture.release()
        writer.release()

    tracked_vehicle_ids = {
        track_id
        for track_id, label in track_classes.items()
        if label in VEHICLE_CLASSES
    }

    average_vehicles_per_frame = (
        total_frame_vehicle_count / processed_frames
        if processed_frames > 0
        else 0.0
    )

    overall_congestion_level = estimate_congestion(
        average_vehicles_per_frame
    )

    average_motion = (
        sum(motion_scores) / len(motion_scores)
        if motion_scores
        else 0.0
    )

    peak_motion = max(motion_scores, default=0.0)

    summary = {
        "video": video_path.name,
        "model": MODEL_PATH.name,
        "tracker": "ByteTrack",
        "confidence_threshold": confidence,
        "image_size": image_size,
        "video_fps": round(fps, 3),
        "input_frame_count": frame_count,
        "processed_frames": processed_frames,
        "frame_width": width,
        "frame_height": height,
        "virtual_line_y": line_y,
        "virtual_line_position_percent": line_position,
        "total_detections_across_frames": total_detections,
        "unique_tracked_objects": len(track_classes),
        "unique_tracked_vehicles": len(tracked_vehicle_ids),
        "total_vehicle_line_crossings": len(crossed_ids),
        "crossings_by_direction": dict(direction_counts),
        "crossings_by_vehicle_class": dict(crossing_class_counts),
        "class_counts_across_frames": dict(
            sorted(class_counts.items())
        ),
        "road_relevant_class_counts": filter_road_relevant_class_counts(
            class_counts
        ),
        "optical_flow_analysis": {
            "enabled": enable_optical_flow,
            "method": "Farneback dense optical flow",
            "analysis_width": flow_width,
            "analysis_height": flow_height,
            "frames_with_motion_measurement": len(motion_scores),
            "average_motion_magnitude": round(
                average_motion,
                4,
            ),
            "peak_motion_magnitude": round(
                peak_motion,
                4,
            ),
            "unit": "pixels per frame at analysis resolution",
            "interpretation": (
                "Measures apparent frame-to-frame pixel motion. "
                "It is not a direct measure of vehicle speed "
                "or congestion."
            ),
        },
        "congestion_estimation": {
            "method": (
                "Heuristic based on detected vehicles per frame"
            ),
            "vehicle_classes_included": sorted(VEHICLE_CLASSES),
            "low_threshold_max_vehicles": LOW_CONGESTION_MAX,
            "moderate_threshold_max_vehicles": MODERATE_CONGESTION_MAX,
            "average_vehicles_per_frame": round(
                average_vehicles_per_frame,
                2,
            ),
            "peak_vehicles_in_one_frame": peak_frame_vehicle_count,
            "estimated_overall_level": overall_congestion_level,
            "frames_by_level": {
                level: congestion_level_counts.get(level, 0)
                for level in ["Low", "Moderate", "High"]
            },
            "interpretation": (
                "Low: 0-5 vehicles per frame; "
                "Moderate: 6-12; High: more than 12. "
                "These thresholds are configurable heuristics, "
                "not calibrated traffic congestion measurements."
            ),
        },
        "output_video": output_video.name,
        "frames": frame_summaries,
        "limitations": [
            (
                "A vehicle is counted only if it crosses the "
                "virtual line during the clip."
            ),
            (
                "Each track ID is counted at most once, "
                "even if it crosses again."
            ),
            (
                "Tracking ID changes can cause missed or "
                "duplicate real-world counts."
            ),
            (
                "The line position may need adjustment for "
                "different camera views."
            ),
            (
                "Congestion levels use vehicle detections per frame, "
                "not calibrated road occupancy or traffic speed."
            ),
            (
                "Optical flow measures apparent pixel motion and "
                "can be affected by camera movement, shadows, "
                "lighting, and moving backgrounds."
            ),
            (
                "Camera angle, occlusion, detection errors, "
                "and scene size can affect congestion estimates."
            ),
            "This is not a calibrated traffic measurement system.",
        ],
    }

    output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return output_video, output_json, summary
