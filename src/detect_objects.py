from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "samples" / "road.jpg"
OUTPUT_PATH = ROOT / "outputs" / "road_detected.jpg"


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Road image not found: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("OpenCV could not read the road image.")

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("Detecting objects...")
    results = model.predict(
        source=image,
        conf=0.10,
        imgsz=960,
        verbose=False,
    )

    annotated_image = results[0].plot()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated_image):
        raise RuntimeError("Could not save the detection image.")

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Objects detected: {len(results[0].boxes)}")
    print(f"Detection image saved to: {OUTPUT_PATH}")

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]
        print(f"{label}: {confidence:.2f}")


if __name__ == "__main__":
    main()