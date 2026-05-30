"""Graph candidate detector.

This detector extracts a same-floor centerline graph from StructureDetector's
walkable area. It does not calculate routes. It returns center nodes and center
edges that stay on the walkable-area skeleton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import numpy as np
from skimage.morphology import skeletonize as skimage_skeletonize

from app.schemas.analyze import Detection
from app.services.base import Detector


@dataclass(frozen=True)
class GraphNode:
    """Internal graph node placed on the walkable centerline."""

    node_id: int
    x: float
    y: float
    label: str
    ocr_text: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    """Internal graph edge following the walkable centerline."""

    start_id: int
    end_id: int
    path: list[list[float]]
    label: str = "centerline_walkway"


class GraphDetector(Detector):
    """Extract centerline node and edge candidates from walkable areas."""

    name = "graph"
    version = "v0.4-centerline"

    NODE_CONFIDENCE = 0.72
    EDGE_CONFIDENCE = 0.70
    POI_ACCESS_CONFIDENCE = 0.70
    CONNECTOR_MERGE_DISTANCE_PX = 3.0
    MIN_EDGE_LENGTH_PX = 18.0
    TURN_ANGLE_DEGREES = 80.0
    PATH_SAMPLE_STEP_PX = 6

    def detect(
        self,
        image_path: str,
        object_detections: List[Detection] | None = None,
        structure_detections: List[Detection] | None = None,
        poi_detections: List[Detection] | None = None,
    ) -> List[Detection]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

        walkable_mask = self._build_walkable_mask(image.shape[:2], structure_detections or [])
        if cv2.countNonZero(walkable_mask) == 0:
            return []

        skeleton = self._skeletonize(walkable_mask)
        if cv2.countNonZero(skeleton) == 0:
            return []

        raw_nodes, raw_node_map = self._extract_raw_nodes(skeleton)
        raw_paths = self._trace_raw_paths(skeleton, raw_nodes, raw_node_map)
        nodes, edges = self._split_paths_into_center_graph(raw_nodes, raw_paths)
        nodes, edges = self._connect_poi_access_nodes(
            walkable_mask,
            nodes,
            edges,
            poi_detections or [],
        )

        return self._to_detections(nodes, edges)

    def _build_walkable_mask(
        self,
        image_shape: tuple[int, int],
        structure_detections: list[Detection],
    ) -> cv2.typing.MatLike:
        height, width = image_shape
        mask = np.zeros((height, width), dtype="uint8")
        blocked_mask = np.zeros((height, width), dtype="uint8")

        for detection in structure_detections:
            geom = detection.geom_px
            if geom.get("type") != "Polygon":
                continue

            coordinates = geom.get("coordinates")
            if not coordinates or not coordinates[0] or len(coordinates[0]) < 3:
                continue

            polygon = np.array(coordinates[0], dtype=np.int32)
            if detection.label == "walkable_area":
                cv2.fillPoly(mask, [polygon], 255)
            elif detection.label in {"blocked_area", "room_area"}:
                cv2.fillPoly(blocked_mask, [polygon], 255)

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(blocked_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _skeletonize(self, mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(mask)

        skeleton = skimage_skeletonize(mask > 0)
        if skeleton.any():
            return skeleton.astype("uint8") * 255

        return self._zhang_suen_thinning(mask)

    @staticmethod
    def _zhang_suen_thinning(mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
        x, y, width, height = cv2.boundingRect(mask)
        if width == 0 or height == 0:
            return np.zeros_like(mask, dtype="uint8")

        roi = (mask[y:y + height, x:x + width] > 0).astype(np.uint8)
        changed = True

        while changed:
            changed = False
            for step in (0, 1):
                padded = np.pad(roi, 1, mode="constant")
                p2 = padded[:-2, 1:-1]
                p3 = padded[:-2, 2:]
                p4 = padded[1:-1, 2:]
                p5 = padded[2:, 2:]
                p6 = padded[2:, 1:-1]
                p7 = padded[2:, :-2]
                p8 = padded[1:-1, :-2]
                p9 = padded[:-2, :-2]

                neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                transitions = (
                    ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                    + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                    + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                    + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                    + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                    + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                    + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                    + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
                )

                if step == 0:
                    preserve = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
                else:
                    preserve = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

                delete = (
                    (roi == 1)
                    & (neighbor_count >= 2)
                    & (neighbor_count <= 6)
                    & (transitions == 1)
                    & preserve
                )
                if delete.any():
                    roi[delete] = 0
                    changed = True

        skeleton = np.zeros_like(mask, dtype="uint8")
        skeleton[y:y + height, x:x + width] = roi * 255
        return skeleton

    def _extract_raw_nodes(
        self,
        skeleton: cv2.typing.MatLike,
    ) -> tuple[list[GraphNode], cv2.typing.MatLike]:
        skeleton_bool = skeleton > 0
        degree = self._neighbor_degree(skeleton_bool)
        candidate_mask = (skeleton_bool & (degree != 2)).astype("uint8") * 255

        count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask)
        node_map = np.zeros_like(skeleton, dtype=np.int32)
        nodes: list[GraphNode] = []

        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] <= 0:
                continue

            ys, xs = np.where(labels == index)
            node_id = len(nodes) + 1
            node_map[ys, xs] = node_id

            max_degree = int(degree[ys, xs].max()) if len(xs) else 0
            label = "center_junction" if max_degree >= 3 else "center_endpoint"
            nodes.append(
                GraphNode(
                    node_id=node_id,
                    x=float(xs.mean()),
                    y=float(ys.mean()),
                    label=label,
                )
            )

        return nodes, node_map

    @staticmethod
    def _neighbor_degree(skeleton_bool: cv2.typing.MatLike) -> cv2.typing.MatLike:
        padded = np.pad(skeleton_bool.astype(np.uint8), 1)
        degree = np.zeros_like(skeleton_bool, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                degree += padded[
                    1 + dy:1 + dy + skeleton_bool.shape[0],
                    1 + dx:1 + dx + skeleton_bool.shape[1],
                ]
        return degree

    def _trace_raw_paths(
        self,
        skeleton: cv2.typing.MatLike,
        nodes: list[GraphNode],
        node_map: cv2.typing.MatLike,
    ) -> list[tuple[int, int, list[tuple[int, int]]]]:
        node_pixels = self._node_pixels(node_map)
        visited_steps: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        seen_node_pairs: set[tuple[int, int, tuple[int, int], tuple[int, int]]] = set()
        paths: list[tuple[int, int, list[tuple[int, int]]]] = []

        for start_id, pixels in node_pixels.items():
            for start_pixel in pixels:
                for neighbor in self._skeleton_neighbors(skeleton, start_pixel):
                    if node_map[neighbor[1], neighbor[0]] == start_id:
                        continue

                    traced = self._trace_path(
                        skeleton=skeleton,
                        node_map=node_map,
                        start_id=start_id,
                        start_pixel=start_pixel,
                        first_pixel=neighbor,
                        visited_steps=visited_steps,
                    )
                    if traced is None:
                        continue

                    end_id, path_pixels = traced
                    start_key = path_pixels[0] if path_pixels else start_pixel
                    end_key = path_pixels[-1] if path_pixels else neighbor
                    pair_key = (min(start_id, end_id), max(start_id, end_id), start_key, end_key)
                    if pair_key in seen_node_pairs:
                        continue

                    seen_node_pairs.add(pair_key)
                    paths.append((start_id, end_id, path_pixels))

        return paths

    @staticmethod
    def _node_pixels(node_map: cv2.typing.MatLike) -> dict[int, list[tuple[int, int]]]:
        node_pixels: dict[int, list[tuple[int, int]]] = {}
        for node_id in np.unique(node_map):
            if node_id == 0:
                continue

            ys, xs = np.where(node_map == node_id)
            node_pixels[int(node_id)] = [
                (int(x), int(y))
                for x, y in zip(xs, ys)
            ]
        return node_pixels

    def _trace_path(
        self,
        skeleton: cv2.typing.MatLike,
        node_map: cv2.typing.MatLike,
        start_id: int,
        start_pixel: tuple[int, int],
        first_pixel: tuple[int, int],
        visited_steps: set[tuple[tuple[int, int], tuple[int, int]]],
    ) -> tuple[int, list[tuple[int, int]]] | None:
        previous = start_pixel
        current = first_pixel
        path_pixels: list[tuple[int, int]] = []

        while True:
            step = (previous, current)
            reverse_step = (current, previous)
            if step in visited_steps:
                return None

            visited_steps.add(step)
            visited_steps.add(reverse_step)

            current_node_id = int(node_map[current[1], current[0]])
            if current_node_id and current_node_id != start_id:
                return current_node_id, path_pixels

            path_pixels.append(current)
            next_pixels = [
                pixel
                for pixel in self._skeleton_neighbors(skeleton, current)
                if pixel != previous
            ]
            if not next_pixels:
                return None

            unvisited = [
                pixel
                for pixel in next_pixels
                if (current, pixel) not in visited_steps
            ]
            previous, current = current, unvisited[0] if unvisited else next_pixels[0]

    @staticmethod
    def _skeleton_neighbors(
        skeleton: cv2.typing.MatLike,
        pixel: tuple[int, int],
    ) -> Iterable[tuple[int, int]]:
        x, y = pixel
        height, width = skeleton.shape[:2]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx] > 0:
                    yield nx, ny

    def _split_paths_into_center_graph(
        self,
        raw_nodes: list[GraphNode],
        raw_paths: list[tuple[int, int, list[tuple[int, int]]]],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        raw_node_lookup = {node.node_id: node for node in raw_nodes}
        nodes: list[GraphNode] = []
        node_by_raw_id: dict[int, int] = {}

        def add_node(x: float, y: float, label: str) -> int:
            for node in nodes:
                if math.hypot(node.x - x, node.y - y) < 2.0:
                    return node.node_id

            node_id = len(nodes) + 1
            nodes.append(GraphNode(node_id=node_id, x=x, y=y, label=label))
            return node_id

        for raw_node in raw_nodes:
            node_by_raw_id[raw_node.node_id] = add_node(
                raw_node.x,
                raw_node.y,
                raw_node.label,
            )

        edges: list[GraphEdge] = []
        seen_edges: set[tuple[int, int, int, int, int, int]] = set()
        for start_raw_id, end_raw_id, path_pixels in raw_paths:
            start_raw = raw_node_lookup[start_raw_id]
            end_raw = raw_node_lookup[end_raw_id]
            center_path = [[start_raw.x, start_raw.y]]
            center_path.extend([[float(x), float(y)] for x, y in path_pixels])
            center_path.append([end_raw.x, end_raw.y])

            split_indices = self._split_indices(center_path)
            node_ids = [node_by_raw_id[start_raw_id]]
            for index in split_indices:
                x, y = center_path[index]
                node_ids.append(add_node(x, y, "center_turn"))
            node_ids.append(node_by_raw_id[end_raw_id])

            split_points = [0, *split_indices, len(center_path) - 1]
            for left, right, start_id, end_id in zip(
                split_points,
                split_points[1:],
                node_ids,
                node_ids[1:],
            ):
                if start_id == end_id:
                    continue

                segment = self._sample_path(center_path[left:right + 1])
                if self._path_length(segment) < self.MIN_EDGE_LENGTH_PX:
                    continue

                edge_key = (
                    min(start_id, end_id),
                    max(start_id, end_id),
                    int(round(segment[0][0])),
                    int(round(segment[0][1])),
                    int(round(segment[-1][0])),
                    int(round(segment[-1][1])),
                )
                if edge_key in seen_edges:
                    continue

                seen_edges.add(edge_key)
                edges.append(GraphEdge(start_id=start_id, end_id=end_id, path=segment))

        return self._renumber_connected_graph(nodes, edges)

    def _split_indices(self, path: list[list[float]]) -> list[int]:
        indices: list[int] = []
        distance_since_node = 0.0

        for index in range(2, len(path) - 2):
            previous = path[index - 2]
            current = path[index]
            following = path[index + 2]
            distance_since_node += math.hypot(
                path[index][0] - path[index - 1][0],
                path[index][1] - path[index - 1][1],
            )

            angle = self._turn_angle(previous, current, following)
            if angle >= self.TURN_ANGLE_DEGREES and distance_since_node >= 30.0:
                indices.append(index)
                distance_since_node = 0.0

        return self._deduplicate_indices(indices)

    @staticmethod
    def _turn_angle(
        previous: list[float],
        current: list[float],
        following: list[float],
    ) -> float:
        vector_a = (current[0] - previous[0], current[1] - previous[1])
        vector_b = (following[0] - current[0], following[1] - current[1])
        length_a = math.hypot(*vector_a)
        length_b = math.hypot(*vector_b)
        if length_a == 0.0 or length_b == 0.0:
            return 0.0

        dot = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]
        cosine = max(-1.0, min(1.0, dot / (length_a * length_b)))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _deduplicate_indices(indices: list[int]) -> list[int]:
        deduplicated: list[int] = []
        for index in indices:
            if deduplicated and index - deduplicated[-1] < 8:
                continue
            deduplicated.append(index)
        return deduplicated

    def _sample_path(self, path: list[list[float]]) -> list[list[float]]:
        if len(path) <= self.PATH_SAMPLE_STEP_PX:
            return path

        sampled = path[::self.PATH_SAMPLE_STEP_PX]
        if sampled[-1] != path[-1]:
            sampled.append(path[-1])
        return sampled

    @staticmethod
    def _path_length(points: list[list[float]]) -> float:
        return sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(points, points[1:])
        )

    @staticmethod
    def _nodes_from_edges(
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[GraphNode]:
        connected_ids = {
            node_id
            for edge in edges
            for node_id in (edge.start_id, edge.end_id)
        }
        return [node for node in nodes if node.node_id in connected_ids]

    def _renumber_connected_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        connected_nodes = self._nodes_from_edges(nodes, edges)
        id_map = {
            node.node_id: index + 1
            for index, node in enumerate(connected_nodes)
        }
        renumbered_nodes = [
            GraphNode(
                node_id=id_map[node.node_id],
                x=node.x,
                y=node.y,
                label=node.label,
                ocr_text=node.ocr_text,
            )
            for node in connected_nodes
        ]
        renumbered_edges = [
            GraphEdge(
                start_id=id_map[edge.start_id],
                end_id=id_map[edge.end_id],
                path=edge.path,
                label=edge.label,
            )
            for edge in edges
            if edge.start_id in id_map and edge.end_id in id_map
        ]
        return renumbered_nodes, renumbered_edges

    def _connect_poi_access_nodes(
        self,
        walkable_mask: cv2.typing.MatLike,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        poi_detections: list[Detection],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if not edges or not poi_detections:
            return nodes, edges

        boundary_points = self._walkable_boundary_points(walkable_mask)
        if len(boundary_points) == 0:
            return nodes, edges

        for poi in poi_detections:
            poi_point = self._point_coordinates(poi)
            if poi.detect_type != "poi_candidate" or poi_point is None:
                continue

            access_point = self._nearest_walkable_point(
                walkable_mask,
                boundary_points,
                poi_point,
            )
            edge_index, connector_point, segment_index = self._nearest_edge_projection(
                walkable_mask,
                edges,
                access_point,
            )
            if edge_index is None or connector_point is None or segment_index is None:
                continue

            connector_id, edges = self._split_edge_at_connector(
                nodes,
                edges,
                edge_index,
                connector_point,
                segment_index,
            )
            access_id = self._add_graph_node(
                nodes,
                access_point,
                label=f"poi_access_node:{poi.label or 'unknown'}",
                ocr_text=poi.ocr_text,
            )
            if access_id == connector_id:
                continue

            edges.append(
                GraphEdge(
                    start_id=access_id,
                    end_id=connector_id,
                    path=[access_point, connector_point],
                    label=f"poi_access_link:{poi.label or 'unknown'}",
                )
            )

        return self._renumber_connected_graph(nodes, edges)

    @staticmethod
    def _walkable_boundary_points(
        walkable_mask: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded = cv2.erode(walkable_mask, kernel)
        boundary = cv2.subtract(walkable_mask, eroded)
        points = cv2.findNonZero(boundary)
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        return points.reshape((-1, 2)).astype(np.float32)

    @staticmethod
    def _nearest_walkable_point(
        walkable_mask: cv2.typing.MatLike,
        boundary_points: cv2.typing.MatLike,
        poi_point: list[float],
    ) -> list[float]:
        x = int(round(poi_point[0]))
        y = int(round(poi_point[1]))
        height, width = walkable_mask.shape[:2]
        if 0 <= x < width and 0 <= y < height and walkable_mask[y, x] > 0:
            return [float(x), float(y)]

        distances = (
            (boundary_points[:, 0] - poi_point[0]) ** 2
            + (boundary_points[:, 1] - poi_point[1]) ** 2
        )
        nearest = boundary_points[int(np.argmin(distances))]
        return [float(nearest[0]), float(nearest[1])]

    def _nearest_edge_projection(
        self,
        walkable_mask: cv2.typing.MatLike,
        edges: list[GraphEdge],
        point: list[float],
    ) -> tuple[int | None, list[float] | None, int | None]:
        nearest: tuple[float, int, list[float], int] | None = None

        for edge_index, edge in enumerate(edges):
            if edge.label.startswith("poi_access_link"):
                continue

            for segment_index, (start, end) in enumerate(zip(edge.path, edge.path[1:])):
                projection = self._project_point_to_segment(point, start, end)
                if not self._line_is_walkable(walkable_mask, point, projection):
                    continue

                distance = math.dist(point, projection)
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, edge_index, projection, segment_index)

        if nearest is None:
            return None, None, None
        return nearest[1], nearest[2], nearest[3]

    @staticmethod
    def _line_is_walkable(
        walkable_mask: cv2.typing.MatLike,
        start: list[float],
        end: list[float],
    ) -> bool:
        line_mask = np.zeros_like(walkable_mask, dtype="uint8")
        cv2.line(
            line_mask,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            255,
            thickness=1,
        )
        line_pixels = line_mask > 0
        return bool(line_pixels.any() and (walkable_mask[line_pixels] > 0).all())

    @staticmethod
    def _project_point_to_segment(
        point: list[float],
        start: list[float],
        end: list[float],
    ) -> list[float]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0.0:
            return [float(start[0]), float(start[1])]

        ratio = (
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / length_squared
        )
        ratio = max(0.0, min(1.0, ratio))
        return [
            float(start[0] + ratio * dx),
            float(start[1] + ratio * dy),
        ]

    def _split_edge_at_connector(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        edge_index: int,
        connector_point: list[float],
        segment_index: int,
    ) -> tuple[int, list[GraphEdge]]:
        edge = edges[edge_index]
        start_node = self._node_by_id(nodes, edge.start_id)
        end_node = self._node_by_id(nodes, edge.end_id)

        if math.dist([start_node.x, start_node.y], connector_point) <= self.CONNECTOR_MERGE_DISTANCE_PX:
            return start_node.node_id, edges
        if math.dist([end_node.x, end_node.y], connector_point) <= self.CONNECTOR_MERGE_DISTANCE_PX:
            return end_node.node_id, edges

        connector_id = self._add_graph_node(nodes, connector_point, "center_connector")
        left_path = [
            *edge.path[:segment_index + 1],
            connector_point,
        ]
        right_path = [
            connector_point,
            *edge.path[segment_index + 1:],
        ]
        replacement_edges = [
            GraphEdge(
                start_id=edge.start_id,
                end_id=connector_id,
                path=left_path,
                label=edge.label,
            ),
            GraphEdge(
                start_id=connector_id,
                end_id=edge.end_id,
                path=right_path,
                label=edge.label,
            ),
        ]
        return connector_id, [
            *edges[:edge_index],
            *replacement_edges,
            *edges[edge_index + 1:],
        ]

    @staticmethod
    def _node_by_id(nodes: list[GraphNode], node_id: int) -> GraphNode:
        for node in nodes:
            if node.node_id == node_id:
                return node
        raise ValueError(f"Graph node not found: {node_id}")

    def _add_graph_node(
        self,
        nodes: list[GraphNode],
        point: list[float],
        label: str,
        ocr_text: str | None = None,
    ) -> int:
        for node in nodes:
            if (
                node.label == label
                and math.dist([node.x, node.y], point) <= self.CONNECTOR_MERGE_DISTANCE_PX
            ):
                return node.node_id

        node_id = max((node.node_id for node in nodes), default=0) + 1
        nodes.append(
            GraphNode(
                node_id=node_id,
                x=float(point[0]),
                y=float(point[1]),
                label=label,
                ocr_text=ocr_text,
            )
        )
        return node_id

    @staticmethod
    def _point_coordinates(detection: Detection) -> list[float] | None:
        if detection.geom_px.get("type") != "Point":
            return None

        coordinates = detection.geom_px.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None
        return [float(coordinates[0]), float(coordinates[1])]

    def _to_detections(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[Detection]:
        detections: list[Detection] = []

        for node in nodes:
            detections.append(
                Detection(
                    detect_type="node_candidate",
                    confidence=self.NODE_CONFIDENCE,
                    geom_px={
                        "type": "Point",
                        "coordinates": [node.x, node.y],
                    },
                    bbox_px=[node.x - 3, node.y - 3, 6, 6],
                    label=node.label,
                    ocr_text=node.ocr_text,
                )
            )

        for edge in edges:
            xs = [point[0] for point in edge.path]
            ys = [point[1] for point in edge.path]
            detections.append(
                Detection(
                    detect_type="edge_candidate",
                    confidence=self.EDGE_CONFIDENCE,
                    geom_px={
                        "type": "LineString",
                        "coordinates": edge.path,
                    },
                    bbox_px=[
                        min(xs),
                        min(ys),
                        max(xs) - min(xs),
                        max(ys) - min(ys),
                    ],
                    label=edge.label,
                )
            )

        return detections
