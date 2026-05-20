"""Structure area detector.

This detector extracts coarse walkable and blocked area candidates from a
floorplan image using OpenCV thresholding. It is intended as the first
rule-based structure extraction step before wall/boundary refinement.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np

from app.schemas.analyze import Detection
from app.services.base import Detector

logger = logging.getLogger(__name__)


class StructureDetector(Detector):
    """Extract walkable and blocked area candidates from floorplan images."""

    name = "structure"
    version = "v0.1-area-threshold"

    WALKABLE_THRESHOLD = 245
    WALL_LINE_THRESHOLD = 150
    WALL_DILATE_KERNEL_SIZE = 7
    BLOCKED_MIN_THRESHOLD = 180
    BLOCKED_MAX_THRESHOLD = 244
    CONTENT_THRESHOLD = 245
    CONTENT_MIN_AREA_RATIO = 0.001
    CONTENT_PADDING_PX = 20
    MIN_AREA_RATIO = 0.0005
    ROOM_AREA_MIN_RATIO = 0.00008
    ROOM_AREA_MIN_WIDTH_PX = 24
    ROOM_AREA_MIN_HEIGHT_PX = 24
    ENCLOSED_BLOCKED_MIN_AREA_RATIO = 0.0002
    APPROX_EPSILON_RATIO = 0.003

    def detect(
        self,
        image_path: str,
        text_detections: list[Detection] | None = None,
        object_detections: list[Detection] | None = None,
    ) -> List[Detection]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = float(gray.shape[0] * gray.shape[1])
        min_area = max(500.0, image_area * self.MIN_AREA_RATIO)

        content_mask = self._build_content_extent_mask(gray)
        walkable_mask, enclosed_blocked_mask = self._build_walkable_masks(gray)
        base_blocked_mask = cv2.bitwise_and(
            self._build_blocked_mask(gray),
            content_mask,
        )
        blocked_mask = cv2.bitwise_or(base_blocked_mask, enclosed_blocked_mask)
        room_area_detections = self._detect_room_areas(
            gray,
            content_mask,
            enclosed_blocked_mask,
            text_detections or [],
            object_detections or [],
            min_area=max(120.0, image_area * self.ROOM_AREA_MIN_RATIO),
        )
        outline_detections = self._detect_wall_outline(
            gray,
            content_mask,
        )

        detections = []
        detections.extend(
            self._mask_to_detections(
                walkable_mask,
                detect_type="corridor",
                label="walkable_area",
                confidence=0.70,
                min_area=min_area,
            )
        )
        detections.extend(
            self._mask_to_detections(
                blocked_mask,
                detect_type="room",
                label="blocked_area",
                confidence=0.70,
                min_area=min_area,
            )
        )
        detections.extend(room_area_detections)
        detections.extend(outline_detections)

        logger.info(
            "Detected %d structure areas from %s",
            len(detections),
            image_path,
        )
        return detections

    def _detect_wall_outline(
        self,
        gray: cv2.typing.MatLike,
        content_mask: cv2.typing.MatLike,
    ) -> list[Detection]:
        floorplan_mask = self._build_wall_outline_mask(gray, content_mask)
        contours, _ = cv2.findContours(
            floorplan_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        polygon = self._contour_to_outline_polygon(contour)
        if len(polygon) < 4:
            return []

        x, y, width, height = cv2.boundingRect(contour)
        return [
            Detection(
                detect_type="wall",
                confidence=0.72,
                geom_px={
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
                bbox_px=[
                    float(x),
                    float(y),
                    float(width),
                    float(height),
                ],
                label="wall_outline",
            )
        ]

    def _build_wall_outline_mask(
        self,
        gray: cv2.typing.MatLike,
        content_mask: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        white_mask = (gray >= self.WALKABLE_THRESHOLD).astype("uint8") * 255
        wall_mask = (gray < self.WALL_LINE_THRESHOLD).astype("uint8") * 255
        wall_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.WALL_DILATE_KERNEL_SIZE, self.WALL_DILATE_KERNEL_SIZE),
        )
        wall_mask = cv2.dilate(wall_mask, wall_kernel, iterations=1)
        open_background = cv2.bitwise_and(white_mask, cv2.bitwise_not(wall_mask))
        outside_background = self._flood_fill_border(open_background)

        floorplan_mask = cv2.bitwise_and(
            cv2.bitwise_not(outside_background),
            content_mask,
        )
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        floorplan_mask = cv2.morphologyEx(
            floorplan_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2,
        )
        return self._keep_largest_component(floorplan_mask)

    @staticmethod
    def _contour_to_outline_polygon(
        contour: cv2.typing.MatLike,
    ) -> list[list[float]]:
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(
            contour,
            epsilon=max(3.0, perimeter * 0.004),
            closed=True,
        )
        polygon = [
            [float(point[0][0]), float(point[0][1])]
            for point in approx
        ]
        if polygon and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        return polygon

    def _detect_room_areas(
        self,
        gray: cv2.typing.MatLike,
        content_mask: cv2.typing.MatLike,
        enclosed_blocked_mask: cv2.typing.MatLike,
        text_detections: list[Detection],
        object_detections: list[Detection],
        min_area: float,
    ) -> list[Detection]:
        room_mask = cv2.bitwise_and(
            self._build_room_area_mask(gray),
            content_mask,
        )
        room_mask = cv2.bitwise_or(room_mask, enclosed_blocked_mask)
        room_detections = self._mask_to_detections(
            room_mask,
            detect_type="room",
            label="room_area",
            confidence=0.68,
            min_area=min_area,
        )
        for room in room_detections:
            room.ocr_text = self._match_room_label(
                room,
                text_detections,
                object_detections,
            )
        return [
            room
            for room in room_detections
            if self._is_room_sized_detection(room)
        ]

    def _build_room_area_mask(self, gray: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = (
            (gray >= self.BLOCKED_MIN_THRESHOLD)
            & (gray <= self.BLOCKED_MAX_THRESHOLD)
        ).astype("uint8") * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _is_room_sized_detection(self, detection: Detection) -> bool:
        if not detection.bbox_px:
            return False

        _, _, width, height = detection.bbox_px
        return (
            width >= self.ROOM_AREA_MIN_WIDTH_PX
            and height >= self.ROOM_AREA_MIN_HEIGHT_PX
        )

    @staticmethod
    def _match_room_label(
        room: Detection,
        text_detections: list[Detection],
        object_detections: list[Detection],
    ) -> str | None:
        texts = [
            detection.ocr_text
            for detection in text_detections
            if detection.ocr_text
            and StructureDetector._detection_center_in_detection(detection, room)
        ]
        if texts:
            return " ".join(texts)

        for detection in object_detections:
            if detection.label and StructureDetector._detection_center_in_detection(
                detection,
                room,
            ):
                return detection.label
        return None

    @staticmethod
    def _detection_center_in_detection(
        child: Detection,
        parent: Detection,
    ) -> bool:
        if not child.bbox_px or not parent.bbox_px:
            return False

        child_x, child_y, child_width, child_height = child.bbox_px
        parent_x, parent_y, parent_width, parent_height = parent.bbox_px
        center_x = child_x + child_width / 2
        center_y = child_y + child_height / 2
        return (
            parent_x <= center_x <= parent_x + parent_width
            and parent_y <= center_y <= parent_y + parent_height
        )

    def _build_walkable_masks(
        self,
        gray: cv2.typing.MatLike,
    ) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        white_mask = (gray >= self.WALKABLE_THRESHOLD).astype("uint8") * 255
        wall_mask = (gray < self.WALL_LINE_THRESHOLD).astype("uint8") * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.WALL_DILATE_KERNEL_SIZE, self.WALL_DILATE_KERNEL_SIZE),
        )
        wall_mask = cv2.dilate(wall_mask, kernel, iterations=1)

        candidate_mask = cv2.bitwise_and(
            white_mask,
            cv2.bitwise_not(wall_mask),
        )
        outside_mask = self._flood_fill_border(candidate_mask)
        internal_mask = cv2.bitwise_and(
            candidate_mask,
            cv2.bitwise_not(outside_mask),
        )
        walkable_mask, enclosed_blocked_mask = self._split_largest_component(
            internal_mask,
            min_other_area=max(
                200.0,
                gray.shape[0] * gray.shape[1] * self.ENCLOSED_BLOCKED_MIN_AREA_RATIO,
            ),
        )
        return self._clean_mask(walkable_mask), self._clean_mask(enclosed_blocked_mask)

    def _build_blocked_mask(self, gray: cv2.typing.MatLike) -> cv2.typing.MatLike:
        mask = (
            (gray >= self.BLOCKED_MIN_THRESHOLD)
            & (gray <= self.BLOCKED_MAX_THRESHOLD)
        ).astype("uint8") * 255
        return self._clean_mask(mask)

    def _build_content_extent_mask(
        self,
        gray: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        content = (gray < self.CONTENT_THRESHOLD).astype("uint8") * 255
        image_area = gray.shape[0] * gray.shape[1]
        min_component_area = image_area * self.CONTENT_MIN_AREA_RATIO

        count, labels, stats, _ = cv2.connectedComponentsWithStats(content)
        boxes = []
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < min_component_area:
                continue
            x = stats[index, cv2.CC_STAT_LEFT]
            y = stats[index, cv2.CC_STAT_TOP]
            width = stats[index, cv2.CC_STAT_WIDTH]
            height = stats[index, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, x + width, y + height))

        mask = np.zeros_like(gray, dtype="uint8")
        if not boxes:
            mask[:, :] = 255
            return mask

        min_x = max(0, min(box[0] for box in boxes) - self.CONTENT_PADDING_PX)
        min_y = max(0, min(box[1] for box in boxes) - self.CONTENT_PADDING_PX)
        max_x = min(gray.shape[1], max(box[2] for box in boxes) + self.CONTENT_PADDING_PX)
        max_y = min(gray.shape[0], max(box[3] for box in boxes) + self.CONTENT_PADDING_PX)
        mask[min_y:max_y, min_x:max_x] = 255
        return mask

    @staticmethod
    def _clean_mask(mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    @staticmethod
    def _flood_fill_border(mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
        height, width = mask.shape[:2]
        filled = mask.copy()
        flood_mask = np.zeros((height + 2, width + 2), dtype="uint8")

        for x in range(width):
            if filled[0, x] == 255:
                cv2.floodFill(filled, flood_mask, (x, 0), 128)
            if filled[height - 1, x] == 255:
                cv2.floodFill(filled, flood_mask, (x, height - 1), 128)

        for y in range(height):
            if filled[y, 0] == 255:
                cv2.floodFill(filled, flood_mask, (0, y), 128)
            if filled[y, width - 1] == 255:
                cv2.floodFill(filled, flood_mask, (width - 1, y), 128)

        return (filled == 128).astype("uint8") * 255

    @staticmethod
    def _keep_largest_component(mask: cv2.typing.MatLike) -> cv2.typing.MatLike:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return mask

        largest_index = max(
            range(1, count),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )
        return (labels == largest_index).astype("uint8") * 255

    @staticmethod
    def _split_largest_component(
        mask: cv2.typing.MatLike,
        min_other_area: float,
    ) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        walkable_mask = np.zeros_like(mask, dtype="uint8")
        enclosed_blocked_mask = np.zeros_like(mask, dtype="uint8")
        if count <= 1:
            return walkable_mask, enclosed_blocked_mask

        largest_index = max(
            range(1, count),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )

        for index in range(1, count):
            component_mask = labels == index
            if index == largest_index:
                walkable_mask[component_mask] = 255
            elif stats[index, cv2.CC_STAT_AREA] >= min_other_area:
                enclosed_blocked_mask[component_mask] = 255

        return walkable_mask, enclosed_blocked_mask

    def _mask_to_detections(
        self,
        mask: cv2.typing.MatLike,
        detect_type: str,
        label: str,
        confidence: float,
        min_area: float,
    ) -> list[Detection]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            polygon = self._contour_to_polygon(contour)
            if len(polygon) < 4:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    detect_type=detect_type,
                    confidence=confidence,
                    geom_px={
                        "type": "Polygon",
                        "coordinates": [polygon],
                    },
                    bbox_px=[
                        float(x),
                        float(y),
                        float(width),
                        float(height),
                    ],
                    label=label,
                )
            )

        return sorted(
            detections,
            key=lambda detection: detection.bbox_px[2] * detection.bbox_px[3]
            if detection.bbox_px
            else 0.0,
            reverse=True,
        )

    def _contour_to_polygon(
        self,
        contour: cv2.typing.MatLike,
    ) -> list[list[float]]:
        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(2.0, perimeter * self.APPROX_EPSILON_RATIO)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        polygon = [
            [float(point[0][0]), float(point[0][1])]
            for point in approx
        ]
        if polygon and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        return polygon
