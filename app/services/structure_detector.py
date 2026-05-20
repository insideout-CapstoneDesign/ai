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
    ENCLOSED_BLOCKED_MIN_AREA_RATIO = 0.0002
    APPROX_EPSILON_RATIO = 0.003

    def detect(self, image_path: str) -> List[Detection]:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = float(gray.shape[0] * gray.shape[1])
        min_area = max(500.0, image_area * self.MIN_AREA_RATIO)

        content_mask = self._build_content_extent_mask(gray)
        walkable_mask, enclosed_blocked_mask = self._build_walkable_masks(gray)
        blocked_mask = cv2.bitwise_and(
            self._build_blocked_mask(gray),
            content_mask,
        )
        blocked_mask = cv2.bitwise_or(blocked_mask, enclosed_blocked_mask)

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

        logger.info(
            "Detected %d structure areas from %s",
            len(detections),
            image_path,
        )
        return detections

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

        seeds = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
        ]
        for seed in seeds:
            if filled[seed[1], seed[0]] == 255:
                cv2.floodFill(filled, flood_mask, seed, 128)

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
