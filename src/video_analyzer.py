
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

# Heuristic thresholds: adjust for your camera view.
LOW_CONGESTION_MAX = 5
MODERATE_CONGESTION_MAX = 12


def estimate_congestion(vehicle_count):
    if vehicle_count <= LOW_CONGESTION_MAX:
        return "Low"
    if vehicle_count <= MODERATE_CONGESTION_MAX:
        return "Moderate"
    return "High"


def analyze_video(
    video_path,
    confidence=0.25,
    image_size=640,
    line_position=65,
):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    model = YOLO(str(MODEL_PATH))

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

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )[0]

            annotated_frame = result.plot()
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
                                "above" if center_y < line_y else "below"
                            )

                            previous_side = previous_sides.get(track_id)

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

            congestion_level = estimate_congestion(frame_vehicle_count)
            total_frame_vehicle_count += frame_vehicle_count
            peak_frame_vehicle_count = max(
                peak_frame_vehicle_count,
                frame_vehicle_count,
            )
            congestion_level_counts[congestion_level] += 1

            cv2.line(
                annotated_frame,
                (0, line_y),
                (width, line_y),
                (0, 255, 255),
                3,
            )

            cv2.putText(
                annotated_frame,
                f"Virtual Line | Crossings: {len(crossed_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Vehicles: {frame_vehicle_count} | Congestion: {congestion_level}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            writer.write(annotated_frame)

            frame_summaries.append({
                "frame": processed_frames,
                "timestamp_seconds": round(
                    (processed_frames - 1) / fps, 3
                ),
                "detection_count": len(frame_detections),
                "vehicle_count": frame_vehicle_count,
                "congestion_level": congestion_level,
                "line_crossings_this_frame": frame_crossings,
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
        "class_counts_across_frames": dict(sorted(class_counts.items())),
        "congestion_estimation": {
            "method": "Heuristic based on detected vehicles per frame",
            "vehicle_classes_included": sorted(VEHICLE_CLASSES),
            "low_threshold_max_vehicles": LOW_CONGESTION_MAX,
            "moderate_threshold_max_vehicles": MODERATE_CONGESTION_MAX,
            "average_vehicles_per_frame": round(
                average_vehicles_per_frame, 2
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
            "A vehicle is counted only if it crosses the virtual line during the clip.",
            "Each track ID is counted at most once, even if it crosses again.",
            "Tracking ID changes can cause missed or duplicate real-world counts.",
            "The line position may need adjustment for different camera views.",
            "Congestion levels use vehicle detections per frame, not calibrated road occupancy or traffic speed.",
            "Camera angle, occlusion, detection errors, and scene size can affect congestion estimates.",
            "This is not a calibrated traffic measurement system.",
        ],
    }

    output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return output_video, output_json, summary
