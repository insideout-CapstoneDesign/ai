"""POI candidate detector.

This detector converts OCR text detections and object/icon detections into
`poi_candidate` detections that the backend can review or store as POIs.
"""

from __future__ import annotations

import math
from typing import Iterable, List

from app.schemas.analyze import Detection
from app.services.base import Detector


class PoiDetector(Detector):
    """Create POI candidates from OCR text and detected facility icons."""

    name = "poi"
    version = "v0.2-candidate-rules"

    POI_OBJECT_LABELS: dict[str, str] = {
        "accessible_restroom": "facility.accessible_restroom",
        "aed": "facility.aed",
        "atm": "facility.atm",
        "cafe": "facility.cafe",
        "clothing_alteration": "store.clothing_alteration",
        "elevator": "facility.elevator",
        "escalator": "facility.escalator",
        "family_restroom": "facility.family_restroom",
        "infodesk": "facility.infodesk",
        "phone_charging": "facility.phone_charging",
        "restroom_female": "facility.restroom_female",
        "restroom_male": "facility.restroom_male",
        "shoe_repair": "store.shoe_repair",
        "stair": "facility.stair",
        "storage_locker": "facility.storage_locker",
        "subway_station": "facility.subway_station",
        "water_fountain": "facility.water_fountain",
    }
    RESTROOM_CATEGORIES = {
        "facility.accessible_restroom",
        "facility.family_restroom",
        "facility.restroom_female",
        "facility.restroom_male",
    }

    KEYWORD_CATEGORIES: tuple[tuple[str, str], ...] = (
        ("장애인", "facility.accessible_restroom"),
        ("AED", "facility.aed"),
        ("자동제세동기", "facility.aed"),
        ("제세동기", "facility.aed"),
        ("ATM", "facility.atm"),
        ("카페", "facility.cafe"),
        ("커피", "facility.cafe"),
        ("AS", "store.clothing_alteration"),
        ("수선", "store.clothing_alteration"),
        ("구두", "store.shoe_repair"),
        ("엘리베이터", "facility.elevator"),
        ("에스컬레이터", "facility.escalator"),
        ("계단", "facility.stair"),
        ("가족", "facility.family_restroom"),
        ("안내", "facility.infodesk"),
        ("인포", "facility.infodesk"),
        ("충전", "facility.phone_charging"),
        ("여자화장실", "facility.restroom_female"),
        ("여자 화장실", "facility.restroom_female"),
        ("남자화장실", "facility.restroom_male"),
        ("남자 화장실", "facility.restroom_male"),
        ("화장실", "facility.restroom"),
        ("물품보관", "facility.storage_locker"),
        ("보관함", "facility.storage_locker"),
        ("지하철", "facility.subway_station"),
        ("음수대", "facility.water_fountain"),
        ("정수기", "facility.water_fountain"),
    )

    def detect(
        self,
        image_path: str,
        text_detections: List[Detection] | None = None,
        object_detections: List[Detection] | None = None,
    ) -> List[Detection]:
        # Kept for the shared Detector interface; POI extraction uses upstream detections.
        _ = image_path
        texts = text_detections or []
        objects = object_detections or []

        candidates, matched_text_ids = self._from_objects(objects, texts)
        candidates.extend(self._from_texts(texts, matched_text_ids))
        return candidates

    def _from_objects(
        self,
        objects: Iterable[Detection],
        texts: list[Detection],
    ) -> tuple[list[Detection], set[int]]:
        candidates: list[Detection] = []
        matched_text_ids: set[int] = set()

        for detection in objects:
            category = self._category_for_object(detection)
            if category is None:
                continue

            point = self._center_point(detection)
            if point is None:
                continue

            matched_text = self._nearest_text(detection, texts, category)
            confidence = detection.confidence
            if matched_text is not None:
                matched_text_ids.add(id(matched_text))
                confidence = min(
                    1.0,
                    (detection.confidence * 0.7) + (matched_text.confidence * 0.3),
                )

            candidates.append(
                Detection(
                    detect_type="poi_candidate",
                    confidence=confidence,
                    geom_px={
                        "type": "Point",
                        "coordinates": point,
                    },
                    bbox_px=detection.bbox_px,
                    label=category,
                    ocr_text=matched_text.ocr_text if matched_text else None,
                )
            )

        return candidates, matched_text_ids

    def _from_texts(
        self,
        texts: Iterable[Detection],
        excluded_text_ids: set[int],
    ) -> list[Detection]:
        candidates: list[Detection] = []

        for detection in texts:
            if id(detection) in excluded_text_ids:
                continue

            text = self._clean_text(detection.ocr_text)
            if not text:
                continue

            category = self._category_for_text(text)
            if category is None:
                category = "store.unknown"

            point = self._center_point(detection)
            if point is None:
                continue

            candidates.append(
                Detection(
                    detect_type="poi_candidate",
                    confidence=min(1.0, detection.confidence * 0.9),
                    geom_px={
                        "type": "Point",
                        "coordinates": point,
                    },
                    bbox_px=detection.bbox_px,
                    label=category,
                    ocr_text=text,
                )
            )

        return candidates

    def _category_for_object(self, detection: Detection) -> str | None:
        if detection.label in self.POI_OBJECT_LABELS:
            return self.POI_OBJECT_LABELS[detection.label]
        if detection.detect_type in self.POI_OBJECT_LABELS:
            return self.POI_OBJECT_LABELS[detection.detect_type]
        return None

    def _category_for_text(self, text: str) -> str | None:
        normalized = text.upper().replace(" ", "")
        for keyword, category in self.KEYWORD_CATEGORIES:
            if keyword.upper().replace(" ", "") in normalized:
                return category
        return None

    def _nearest_text(
        self,
        source: Detection,
        texts: Iterable[Detection],
        source_category: str,
    ) -> Detection | None:
        source_point = self._center_point(source)
        source_bbox = source.bbox_px
        if source_point is None or source_bbox is None:
            return None

        max_distance = max(80.0, max(source_bbox[2], source_bbox[3]) * 3.0)
        nearest: tuple[float, Detection] | None = None

        for text_detection in texts:
            text = self._clean_text(text_detection.ocr_text)
            if not text:
                continue
            if not self._is_compatible_text_category(source_category, text):
                continue

            text_point = self._center_point(text_detection)
            if text_point is None:
                continue

            distance = math.dist(source_point, text_point)
            if distance > max_distance:
                continue
            if nearest is None or distance < nearest[0]:
                nearest = (distance, text_detection)

        return nearest[1] if nearest else None

    def _is_compatible_text_category(
        self,
        source_category: str,
        text: str,
    ) -> bool:
        text_category = self._category_for_text(text)
        if text_category is None:
            return False
        if text_category == source_category:
            return True
        return (
            text_category == "facility.restroom"
            and source_category in self.RESTROOM_CATEGORIES
        )

    @staticmethod
    def _clean_text(text: str | None) -> str:
        return " ".join((text or "").split())

    @classmethod
    def _center_point(cls, detection: Detection) -> list[float] | None:
        if detection.bbox_px:
            x, y, width, height = detection.bbox_px
            return [x + width / 2, y + height / 2]
        return cls._point_coordinates(detection)

    @staticmethod
    def _point_coordinates(detection: Detection) -> list[float] | None:
        if detection.geom_px.get("type") != "Point":
            return None
        coordinates = detection.geom_px.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            return None
        return [float(coordinates[0]), float(coordinates[1])]
