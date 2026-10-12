
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs" / "video_detection_summary.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation_report.md"


def load_report(path):
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    required = [
        "video",
        "model",
        "tracker",
        "processed_frames",
        "total_detections_across_frames",
        "unique_tracked_objects",
        "unique_tracked_vehicles",
        "total_vehicle_line_crossings",
        "class_counts_across_frames",
        "optical_flow_analysis",
        "congestion_estimation",
        "limitations",
    ]

    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f"Missing required report fields: {', '.join(missing)}")

    return report


def build_evaluation(report):
    optical = report["optical_flow_analysis"]
    congestion = report["congestion_estimation"]
    class_counts = report["class_counts_across_frames"]
    crossings = report.get("crossings_by_vehicle_class", {})
    direction_counts = report.get("crossings_by_direction", {})

    lines = [
        "# AI RoadGuard Evaluation Report",
        "",
        "## Run Configuration",
        "",
        f"- Input video: `{report['video']}`",
        f"- Model: `{report['model']}`",
        f"- Tracker: `{report['tracker']}`",
        f"- Confidence threshold: {report.get('confidence_threshold', 'Not recorded')}",
        f"- Inference image size: {report.get('image_size', 'Not recorded')}",
        f"- Processed frames: {report['processed_frames']}",
        f"- Input frames: {report.get('input_frame_count', 'Not recorded')}",
        f"- Frame dimensions: {report.get('frame_width', '?')} x {report.get('frame_height', '?')}",
        f"- Video FPS: {report.get('video_fps', 'Not recorded')}",
        "",
        "## Recorded Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Detections across frames | {report['total_detections_across_frames']} |",
        f"| Unique tracked objects | {report['unique_tracked_objects']} |",
        f"| Unique tracked vehicles | {report['unique_tracked_vehicles']} |",
        f"| Virtual-line crossings | {report['total_vehicle_line_crossings']} |",
        "",
        "### Detected Classes",
        "",
        "| Class | Detection count across frames |",
        "|---|---:|",
    ]

    for name, count in sorted(
        class_counts.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "### Vehicle-Line Crossings",
        "",
        "| Vehicle class | Count |",
        "|---|---:|",
    ])

    for name, count in sorted(crossings.items()):
        lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "Crossings by direction:",
        "",
    ])

    if direction_counts:
        for direction, count in sorted(direction_counts.items()):
            lines.append(f"- {direction}: {count}")
    else:
        lines.append("- No directional counts recorded.")

    lines.extend([
        "",
        "### Optical Flow",
        "",
        f"- Enabled: {optical.get('enabled', 'Not recorded')}",
        f"- Method: {optical.get('method', 'Not recorded')}",
        f"- Analysis resolution: {optical.get('analysis_width', '?')} x "
        f"{optical.get('analysis_height', '?')}",
        f"- Frames with measurements: {optical.get('frames_with_motion_measurement', 'Not recorded')}",
        f"- Average motion magnitude: {optical.get('average_motion_magnitude', 'Not recorded')}",
        f"- Peak motion magnitude: {optical.get('peak_motion_magnitude', 'Not recorded')}",
        f"- Unit: {optical.get('unit', 'Not recorded')}",
        "",
        "Optical flow represents apparent pixel motion at the analysis resolution. "
        "It is not a direct measurement of vehicle speed.",
        "",
        "### Congestion Heuristic",
        "",
        f"- Method: {congestion.get('method', 'Not recorded')}",
        f"- Average detected vehicles per frame: {congestion.get('average_vehicles_per_frame', 'Not recorded')}",
        f"- Peak detected vehicles in one frame: {congestion.get('peak_vehicles_in_one_frame', 'Not recorded')}",
        f"- Estimated overall level: **{congestion.get('estimated_overall_level', 'Not recorded')}**",
        "",
        "| Estimated level | Frames |",
        "|---|---:|",
    ])

    for level, count in congestion.get("frames_by_level", {}).items():
        lines.append(f"| {level} | {count} |")

    lines.extend([
        "",
        "These traffic levels are heuristic outputs based on detected vehicle counts. "
        "They are not calibrated measurements of congestion.",
        "",
        "## Limitations",
        "",
    ])

    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend([
        "",
        "## Evaluation Status",
        "",
        "- This report summarizes one recorded run.",
        "- Detection counts are not detection-accuracy metrics.",
        "- Precision and recall require ground-truth object annotations.",
        "- Tracking accuracy requires ground-truth identities or a suitable benchmark.",
        "- Crossing-count accuracy requires manually verified crossing counts.",
        "- Results should be checked on additional videos with different lighting, "
        "camera angles, traffic density, and occlusion.",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Create a Markdown evaluation report from AI RoadGuard JSON."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        report = load_report(args.input)
        markdown = build_evaluation(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Evaluation report saved to: {args.output}")
    print(f"Processed frames: {report['processed_frames']}")
    print(f"Tracked vehicles: {report['unique_tracked_vehicles']}")
    print(f"Line crossings: {report['total_vehicle_line_crossings']}")


if __name__ == "__main__":
    main()
