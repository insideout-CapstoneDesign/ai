"""Draw StructureDetector results on a floorplan image.

Usage:
    python scripts/visualize_structure_detections.py input.png output.png
    python scripts/visualize_structure_detections.py input.png runs/structure --split
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
    "room_area": (0, 165, 255),
    "wall_outline": (255, 0, 0),
}

MODE_LABELS = {
    "all": {
        "walkable_area",
        "blocked_area",
        "room_area",
        "wall_outline",
    },
    "areas": {"walkable_area", "blocked_area"},
    "rooms": {"room_area"},
    "outline": {"wall_outline"},
}

LEGEND_ITEMS = {
    "walkable_area": "walkable_area: movable corridor",
    "blocked_area": "blocked_area: non-walkable room/store",
    "room_area": "room_area: individual room/store",
    "wall_outline": "wall_outline: outer map boundary",
}

LABEL_ORDER = (
    "walkable_area",
    "blocked_area",
    "room_area",
    "wall_outline",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize OpenCV structure area detections.",
    )
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_path", help="Output preview image path")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_LABELS),
        default="all",
        help="Layer to visualize when --split is not used",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Write all, areas, wall, and room preview files separately",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    detections = StructureDetector().detect(str(image_path))
    if args.split:
        output_stem = output_path.with_suffix("") if output_path.suffix else output_path
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        for mode in ("areas", "rooms", "outline", "all"):
            mode_output = output_stem.with_name(f"{output_stem.name}_{mode}.png")
            _write_preview(image, detections, mode, mode_output)
        _print_summary(detections, output_stem)
        return

    _write_preview(image, detections, args.mode, output_path)
    _print_summary(detections, output_path)


def _write_preview(
    image: cv2.typing.MatLike,
    detections: list,
    mode: str,
    output_path: Path,
) -> None:
    selected_labels = MODE_LABELS[mode]
    canvas = image.copy()
    overlay = canvas.copy()

    for detection in detections:
        if detection.label not in selected_labels:
            continue

        color = COLORS.get(detection.label or "", (255, 255, 0))
        coordinates = detection.geom_px.get("coordinates")
        if not coordinates:
            continue

        if detection.geom_px.get("type") == "Polygon":
            polygon = np.array(coordinates[0], dtype=np.int32)
            if detection.label == "wall_outline":
                cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=6)
            else:
                cv2.fillPoly(overlay, [polygon], color)
                cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2)
        elif detection.geom_px.get("type") == "LineString":
            line = np.array(coordinates, dtype=np.int32)
            cv2.polylines(canvas, [line], isClosed=False, color=color, thickness=4)

    alpha = 0.28 if mode in {"all", "areas", "rooms"} else 0.08
    blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0)
    _draw_legend(blended, mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), blended)
    if not saved:
        raise IOError(f"Failed to write output image: {output_path}")


def _draw_legend(image: cv2.typing.MatLike, mode: str) -> None:
    labels = [
        label
        for label in LABEL_ORDER
        if label in MODE_LABELS[mode] and label in LEGEND_ITEMS
    ]
    if not labels:
        return

    x = 18
    y = 22
    line_height = 28
    width = 460
    height = 18 + line_height * len(labels)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (255, 255, 255), -1)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (60, 60, 60), 1)

    for index, label in enumerate(labels):
        item_y = y + index * line_height
        color = COLORS[label]
        cv2.line(image, (x, item_y), (x + 32, item_y), color, 5)
        cv2.putText(
            image,
            LEGEND_ITEMS[label],
            (x + 42, item_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def _print_summary(detections: list, output_path: Path) -> None:
    walkable_count = sum(1 for d in detections if d.label == "walkable_area")
    blocked_count = sum(1 for d in detections if d.label == "blocked_area")
    room_area_count = sum(1 for d in detections if d.label == "room_area")
    outline_count = sum(1 for d in detections if d.label == "wall_outline")
    print(
        f"Saved {len(detections)} detections "
        f"({walkable_count} walkable, {blocked_count} blocked, "
        f"{room_area_count} room areas, "
        f"{outline_count} outlines) "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
