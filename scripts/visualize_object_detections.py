"""
Draw ObjectDetector results on a floorplan image.

Usage:
    python scripts/visualize_object_detections.py input.png output.png
"""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.object_detector import ObjectDetector


BOX_COLORS = {
    "elevator": (0, 180, 255),
    "escalator": (255, 120, 0),
    "stair": (80, 200, 120),
    "restroom_sign": (220, 80, 220),
    "poi_candidate": (80, 140, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize OpenCV template matching detections.",
    )
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_path", help="Output preview image path")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    detections = ObjectDetector().detect(str(image_path))

    for detection in detections:
        if detection.bbox_px is None:
            continue

        x, y, width, height = [int(round(value)) for value in detection.bbox_px]
        right = x + width
        bottom = y + height
        color = BOX_COLORS.get(detection.detect_type, (0, 255, 255))
        text = f"{detection.label} {detection.confidence:.2f}"

        cv2.rectangle(image, (x, y), (right, bottom), color, 3)
        cv2.putText(
            image,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved {len(detections)} detections to {output_path}")


if __name__ == "__main__":
    main()
