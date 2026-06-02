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
from app.services.object_detector import ObjectDetector
from app.services.poi_detector import PoiDetector
from app.services.structure_detector import StructureDetector
from app.services.text_detector import TextDetector


EDGE_COLOR = (255, 0, 0)
CENTER_NODE_COLOR = (0, 0, 255)
CONNECTOR_NODE_COLOR = (255, 255, 0)
POI_ACCESS_NODE_COLOR = (255, 0, 255)
POI_ACCESS_EDGE_COLOR = (255, 0, 255)
POI_SOURCE_COLOR = (0, 215, 255)
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
    parser.add_argument(
        "--show-poi-access",
        action="store_true",
        help="Run upstream detectors and draw POI access graph candidates",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    structure_detector = StructureDetector()
    graph_detector = GraphDetector()
    objects = ObjectDetector().detect(str(image_path)) if args.show_poi_access else []
    texts = (
        TextDetector().detect(
            str(image_path),
            object_detections=objects,
        )
        if args.show_poi_access
        else []
    )
    pois = (
        PoiDetector().detect(
            str(image_path),
            text_detections=texts,
            object_detections=objects,
        )
        if args.show_poi_access
        else []
    )
    structures = structure_detector.detect(
        str(image_path),
        text_detections=texts,
        object_detections=objects,
    )
    graph = graph_detector.detect(
        str(image_path),
        structure_detections=structures,
        poi_detections=pois,
    )

    canvas = image.copy()
    if args.show_walkable:
        walkable_mask = graph_detector._build_walkable_mask(image.shape[:2], structures)
        _draw_walkable_area(canvas, walkable_mask)

    _draw_edges(canvas, graph)
    _draw_nodes(canvas, graph)
    if args.show_poi_access:
        _draw_poi_sources(canvas, pois)
    _draw_legend(canvas, args.show_poi_access)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), canvas)
    if not saved:
        raise IOError(f"Failed to write output image: {output_path}")

    node_count = sum(1 for detection in graph if detection.detect_type == "node_candidate")
    edge_count = sum(1 for detection in graph if detection.detect_type == "edge_candidate")
    access_node_count = sum(
        1
        for detection in graph
        if (detection.label or "").startswith("poi_access_node")
    )
    connector_count = sum(
        1
        for detection in graph
        if detection.label == "center_connector"
    )
    access_edge_count = sum(
        1
        for detection in graph
        if (detection.label or "").startswith("poi_access_link")
    )
    print(
        f"Saved {node_count} nodes and {edge_count} edges "
        f"({access_node_count} POI access nodes, "
        f"{connector_count} center connectors, "
        f"{access_edge_count} POI access edges) "
        f"to {output_path}"
    )


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
        color = (
            POI_ACCESS_EDGE_COLOR
            if (detection.label or "").startswith("poi_access_link")
            else EDGE_COLOR
        )
        cv2.polylines(image, [line], isClosed=False, color=color, thickness=3)


def _draw_nodes(image: cv2.typing.MatLike, detections: list) -> None:
    for detection in detections:
        if detection.detect_type != "node_candidate":
            continue

        coordinates = detection.geom_px.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            continue

        x = int(round(coordinates[0]))
        y = int(round(coordinates[1]))
        if (detection.label or "").startswith("poi_access_node"):
            color = POI_ACCESS_NODE_COLOR
        elif detection.label == "center_connector":
            color = CONNECTOR_NODE_COLOR
        else:
            color = CENTER_NODE_COLOR
        cv2.circle(image, (x, y), 6, color, -1)
        cv2.circle(image, (x, y), 8, (255, 255, 255), 2)


def _draw_poi_sources(image: cv2.typing.MatLike, detections: list) -> None:
    for detection in detections:
        coordinates = detection.geom_px.get("coordinates")
        if detection.detect_type != "poi_candidate" or not coordinates or len(coordinates) < 2:
            continue
        x = int(round(coordinates[0]))
        y = int(round(coordinates[1]))
        cv2.drawMarker(
            image,
            (x, y),
            POI_SOURCE_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )


def _draw_legend(image: cv2.typing.MatLike, show_poi_access: bool) -> None:
    x = 18
    y = 24
    line_height = 28
    width = 360
    items = [
        (EDGE_COLOR, "edge_candidate: centerline"),
        (CENTER_NODE_COLOR, "node_candidate: center node"),
    ]
    if show_poi_access:
        items.extend(
            [
                (CONNECTOR_NODE_COLOR, "node_candidate: center connector"),
                (POI_ACCESS_NODE_COLOR, "node_candidate: POI access"),
                (POI_SOURCE_COLOR, "poi_candidate: source"),
            ]
        )

    height = 18 + line_height * len(items)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (255, 255, 255), -1)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (60, 60, 60), 1)

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
