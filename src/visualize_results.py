import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "outputs" / "road_detected.jpg"
REPORT_PATH = ROOT / "outputs" / "hazard_report.json"

OUTPUT_IMAGE = ROOT / "outputs" / "roadguard_demo.jpg"
OUTPUT_PREVIEW = ROOT / "outputs" / "roadguard_preview.jpg"

MAX_ALERTS_TO_DISPLAY = 6


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "Detected image not found. Run: "
            "python src\\detect_objects.py"
        )

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            "Hazard report not found. Run: "
            "python src\\analyze_hazards.py"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    height, width = image.shape[:2]

    # Scale the panel and text for the image resolution.
    scale = max(width / 1920, 0.7)
    margin = int(30 * scale)
    padding = int(24 * scale)
    line_height = int(43 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8 * scale
    thickness = max(1, int(2 * scale))

    alerts = report.get("alerts", [])
    visible_alerts = alerts[:MAX_ALERTS_TO_DISPLAY]

    lines = [
        "ROADGUARD - ROAD SCENE REVIEW",
        f"Detected objects: {report.get('total_detections', 0)}",
        f"Detected vehicles: {report.get('detected_vehicle_count', 0)}",
        f"Review alerts: {report.get('total_alerts', 0)}",
        "Alerts are preliminary; human verification required.",
    ]

    for index, alert in enumerate(visible_alerts, start=1):
        message = alert.get("message", "Review required")
        max_chars = max(30, int(65 / scale))
        if len(message) > max_chars:
            message = message[:max_chars - 3] + "..."

        lines.append(f"{index}. {message}")

    remaining = len(alerts) - len(visible_alerts)
    if remaining > 0:
        lines.append(f"...and {remaining} more alerts in the JSON report.")

    panel_width = min(width - 2 * margin, int(1600 * scale))
    panel_height = padding * 2 + line_height * len(lines)

    # Keep the panel inside the image.
    panel_height = min(panel_height, height - 2 * margin)

    x1 = margin
    y1 = margin
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (25, 25, 25),
        thickness=-1,
    )

    image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0)

    y = y1 + padding + int(24 * scale)

    for index, line in enumerate(lines):
        if y > y2 - padding:
            break

        text_color = (255, 255, 255)

        if index == 0:
            text_color = (0, 220, 255)
        elif line.startswith("Review alerts:"):
            text_color = (0, 165, 255)
        elif line.startswith(tuple(f"{n}." for n in range(1, 10))):
            text_color = (0, 220, 255)

        cv2.putText(
            image,
            line,
            (x1 + padding, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        y += line_height

    if not cv2.imwrite(str(OUTPUT_IMAGE), image):
        raise RuntimeError("Could not save the demo image.")

    # Create a smaller preview for quick viewing and sharing.
    preview_width = min(1280, width)
    preview_height = int(height * preview_width / width)

    preview = cv2.resize(
        image,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )

    if not cv2.imwrite(str(OUTPUT_PREVIEW), preview):
        raise RuntimeError("Could not save the preview image.")

    print("AI RoadGuard visualization completed.")
    print(f"Full-resolution image: {OUTPUT_IMAGE}")
    print(f"Preview image: {OUTPUT_PREVIEW}")


if __name__ == "__main__":
    main()