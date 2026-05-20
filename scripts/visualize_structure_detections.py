"""Draw StructureDetector results on a floorplan image.

Usage:
    python scripts/visualize_structure_detections.py input.png output.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.structure_detector import StructureDetector


COLORS = {
    "walkable_area": (0, 180, 0),
    "blocked_area": (0, 0, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize OpenCV structure area detections.",
    )
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_path", help="Output preview image path")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    detections = StructureDetector().detect(str(image_path))
    overlay = image.copy()

    for detection in detections:
        coordinates = detection.geom_px.get("coordinates")
        if not coordinates:
            continue

        color = COLORS.get(detection.label or "", (255, 255, 0))
        polygon = np.array(coordinates[0], dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

        if detection.bbox_px:
            x, y, width, _ = [int(round(value)) for value in detection.bbox_px]
            text = f"{detection.label} {detection.confidence:.2f}"
            cv2.putText(
                image,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    blended = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)

    walkable_count = sum(1 for d in detections if d.label == "walkable_area")
    blocked_count = sum(1 for d in detections if d.label == "blocked_area")
    print(
        f"Saved {len(detections)} detections "
        f"({walkable_count} walkable, {blocked_count} blocked) to {output_path}"
    )


if __name__ == "__main__":
    main()
