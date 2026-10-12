import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "samples" / "road.jpg"
MODEL_PATH = ROOT / "yolov8n.pt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_IMAGE = OUTPUT_DIR / "road_detected.jpg"
OUTPUT_JSON = OUTPUT_DIR / "detection_summary.json"

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 960


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(f"Could not read image: {IMAGE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(str(MODEL_PATH))

    print("Detecting objects...")
    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    result = results[0]
    annotated_image = result.plot()

    if not cv2.imwrite(str(OUTPUT_IMAGE), annotated_image):
        raise RuntimeError("Could not save annotated image.")

    detections = []
    class_counts = Counter()

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = model.names[class_id]
        coordinates = [round(float(value), 2) for value in box.xyxy[0].tolist()]

        class_counts[label] += 1

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bounding_box": coordinates,
        })

    summary = {
        "image": IMAGE_PATH.name,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "model": MODEL_PATH.name,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "total_detections": len(detections),
        "class_counts": dict(sorted(class_counts.items())),
        "detections": detections,
    }

    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nOpenCV version: {cv2.__version__}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Total detections: {len(detections)}")
    print("\nObject counts:")

    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count}")

    print(f"\nAnnotated image: {OUTPUT_IMAGE}")
    print(f"Detection summary: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()