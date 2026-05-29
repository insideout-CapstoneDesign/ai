"""Draw GraphDetector node and edge candidates on a floorplan image.

Usage:
    python scripts/visualize_graph_detections.py input.png output.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.graph_detector import GraphDetector
from app.services.structure_detector import StructureDetector


EDGE_COLOR = (255, 0, 0)
WAYPOINT_COLOR = (0, 0, 255)
WALKABLE_COLOR = (0, 180, 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize graph node and edge candidates.",
    )
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_path", help="Output preview image path")
    parser.add_argument(
        "--show-walkable",
        action="store_true",
        help="Also draw the walkable area used for skeleton extraction",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    structure_detector = StructureDetector()
    graph_detector = GraphDetector()
    structures = structure_detector.detect(str(image_path))
    graph = graph_detector.detect(str(image_path), structure_detections=structures)

    canvas = image.copy()
    if args.show_walkable:
        walkable_mask = graph_detector._build_walkable_mask(image.shape[:2], structures)
        _draw_walkable_area(canvas, walkable_mask)

    _draw_edges(canvas, graph)
    _draw_nodes(canvas, graph)
    _draw_legend(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), canvas)
    if not saved:
        raise IOError(f"Failed to write output image: {output_path}")

    node_count = sum(1 for detection in graph if detection.detect_type == "node_candidate")
    edge_count = sum(1 for detection in graph if detection.detect_type == "edge_candidate")
    print(f"Saved {node_count} nodes and {edge_count} edges to {output_path}")


def _draw_walkable_area(
    image: cv2.typing.MatLike,
    walkable_mask: cv2.typing.MatLike,
) -> None:
    overlay = image.copy()
    overlay[walkable_mask > 0] = WALKABLE_COLOR
    cv2.addWeighted(overlay, 0.18, image, 0.82, 0, image)


def _draw_edges(image: cv2.typing.MatLike, detections: list) -> None:
    for detection in detections:
        if detection.detect_type != "edge_candidate":
            continue

        coordinates = detection.geom_px.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue

        line = np.array(coordinates, dtype=np.int32)
        cv2.polylines(image, [line], isClosed=False, color=EDGE_COLOR, thickness=3)


def _draw_nodes(image: cv2.typing.MatLike, detections: list) -> None:
    for detection in detections:
        if detection.detect_type != "node_candidate":
            continue

        coordinates = detection.geom_px.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue

        x = int(round(coordinates[0]))
        y = int(round(coordinates[1]))
        cv2.circle(image, (x, y), 6, WAYPOINT_COLOR, -1)
        cv2.circle(image, (x, y), 8, (255, 255, 255), 2)


def _draw_legend(image: cv2.typing.MatLike) -> None:
    x = 18
    y = 24
    line_height = 28
    width = 360
    height = 18 + line_height * 3
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (255, 255, 255), -1)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (60, 60, 60), 1)

    items = (
        (EDGE_COLOR, "edge_candidate: walkway"),
        (WAYPOINT_COLOR, "node_candidate: center node"),
    )
    for index, (color, label) in enumerate(items):
        item_y = y + index * line_height
        cv2.line(image, (x, item_y), (x + 32, item_y), color, 5)
        cv2.putText(
            image,
            label,
            (x + 42, item_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    main()
