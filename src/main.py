
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = ROOT / "samples" / "road.jpg"
OUTPUT_PATH = ROOT / "outputs" / "road_processed.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    height, width = image.shape[:2]

    processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), processed):
        raise RuntimeError("Could not save processed image.")

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Image dimensions: {width} x {height}")
    print(f"Processed image saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()