"""
텍스트(OCR) 추출 Detector.

EasyOCR를 사용해 도면 이미지 안의 텍스트를 추출하고, 이후 POI 추출에서
그대로 활용할 수 있도록 Detection 형태로 반환한다.
"""

import logging
from pathlib import Path
from typing import Any, List

from app.core.config import settings
from app.schemas.analyze import Detection
from app.services.base import Detector

logger = logging.getLogger(__name__)


class TextDetector(Detector):
    """EasyOCR를 사용해 도면의 텍스트 위치와 인식 문자열을 추출한다."""

    name = "text"
    version = "easyocr-1.7.2"

    def __init__(self) -> None:
        self._reader: Any | None = None

    def detect(self, image_path: str) -> List[Detection]:
        """
        이미지에서 텍스트를 추출한다.

        EasyOCR readtext 결과는 (bbox, text, confidence) 형태다.
        bbox는 4개 꼭짓점 좌표이며, 응답에서는 GeoJSON Polygon과
        [x, y, width, height] bbox로 함께 제공한다.
        """
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        reader = self._get_reader()
        raw_results = reader.readtext(image_path)

        detections: List[Detection] = []
        for raw_bbox, raw_text, raw_confidence in raw_results:
            text = str(raw_text).strip()
            confidence = self._clamp_confidence(raw_confidence)
            if not text or confidence < settings.ocr_confidence_threshold:
                continue

            polygon = self._normalize_polygon(raw_bbox)
            bbox = self._polygon_to_bbox(polygon)

            detections.append(
                Detection(
                    detect_type="text",
                    confidence=confidence,
                    geom_px={
                        "type": "Polygon",
                        "coordinates": [polygon + [polygon[0]]],
                    },
                    bbox_px=bbox,
                    label="ocr_text",
                    ocr_text=text,
                )
            )

        detections = self._merge_stacked_texts(detections)
        logger.info("Detected %d text regions from %s", len(detections), image_path)
        return detections

    def _get_reader(self) -> Any:
        """EasyOCR Reader는 무겁기 때문에 최초 요청 시 한 번만 생성한다."""
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR is not installed. Run 'pip install -r requirements.txt'."
                ) from exc

            self._reader = easyocr.Reader(
                self._ocr_languages(),
                gpu=settings.ocr_gpu,
                model_storage_directory=settings.ocr_model_storage_dir,
                download_enabled=settings.ocr_download_enabled,
                verbose=False,
            )
        return self._reader

    @staticmethod
    def _ocr_languages() -> List[str]:
        """
        OCR_LANGUAGE 설정을 EasyOCR 언어 목록으로 변환한다.

        예: "ko" -> ["ko", "en"], "ko,en" -> ["ko", "en"].
        EasyOCR 한국어 인식에는 영어가 함께 포함되어야 한다.
        """
        configured = [
            lang.strip()
            for lang in settings.ocr_language.split(",")
            if lang.strip()
        ]
        if not configured:
            configured = ["ko"]

        if "ko" in configured and "en" not in configured:
            configured.append("en")

        return configured

    @staticmethod
    def _normalize_polygon(raw_bbox: Any) -> List[List[float]]:
        return [
            [float(point[0]), float(point[1])]
            for point in raw_bbox
        ]

    @staticmethod
    def _polygon_to_bbox(polygon: List[List[float]]) -> List[float]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        min_x = min(xs)
        min_y = min(ys)
        return [
            min_x,
            min_y,
            max(xs) - min_x,
            max(ys) - min_y,
        ]

    @staticmethod
    def _clamp_confidence(confidence: Any) -> float:
        return max(0.0, min(1.0, float(confidence)))

    @classmethod
    def _merge_stacked_texts(cls, detections: List[Detection]) -> List[Detection]:
        """
        Merge tightly stacked OCR lines into one text detection.

        Floorplan shop names are sometimes rendered on two lines, e.g.
        "보테가" + "베네타". EasyOCR returns them as separate boxes, so this
        merges only small boxes that are horizontally aligned and almost
        touching vertically. Larger labels and legend rows stay separate.
        """
        unused = sorted(
            detections,
            key=lambda detection: (
                detection.bbox_px[1] if detection.bbox_px else 0,
                detection.bbox_px[0] if detection.bbox_px else 0,
            ),
        )
        merged: List[Detection] = []

        while unused:
            group = [unused.pop(0)]
            changed = True

            while changed:
                changed = False
                for candidate in list(unused):
                    if cls._can_merge_text_group(group, candidate):
                        group.append(candidate)
                        unused.remove(candidate)
                        changed = True
                        break

            merged.append(cls._merge_text_group(group))

        return sorted(
            merged,
            key=lambda detection: (
                detection.bbox_px[1] if detection.bbox_px else 0,
                detection.bbox_px[0] if detection.bbox_px else 0,
            ),
        )

    @classmethod
    def _can_merge_text_group(
        cls,
        group: List[Detection],
        candidate: Detection,
    ) -> bool:
        if candidate.bbox_px is None or not candidate.ocr_text:
            return False
        if any(member.bbox_px is None or not member.ocr_text for member in group):
            return False

        group_bbox = cls._union_bbox([member.bbox_px for member in group])
        candidate_bbox = candidate.bbox_px

        if not cls._is_mergeable_text_box(group_bbox):
            return False
        if not cls._is_mergeable_text_box(candidate_bbox):
            return False

        vertical_gap = candidate_bbox[1] - (group_bbox[1] + group_bbox[3])
        if vertical_gap < -max(group_bbox[3], candidate_bbox[3]) * 0.5:
            return False
        if vertical_gap > min(12.0, max(group_bbox[3], candidate_bbox[3]) * 0.6):
            return False

        overlap_ratio = cls._horizontal_overlap_ratio(group_bbox, candidate_bbox)
        if overlap_ratio < 0.45:
            return False

        group_center_x = group_bbox[0] + group_bbox[2] / 2
        candidate_center_x = candidate_bbox[0] + candidate_bbox[2] / 2
        max_width = max(group_bbox[2], candidate_bbox[2])
        return abs(group_center_x - candidate_center_x) <= max_width * 0.35

    @staticmethod
    def _is_mergeable_text_box(bbox: List[float]) -> bool:
        _, _, width, height = bbox
        return height <= 35 and width <= 130

    @classmethod
    def _merge_text_group(cls, group: List[Detection]) -> Detection:
        if len(group) == 1:
            return group[0]

        ordered = sorted(group, key=lambda detection: detection.bbox_px[1])
        bbox = cls._union_bbox([detection.bbox_px for detection in ordered])
        confidence = sum(detection.confidence for detection in ordered) / len(ordered)
        text = " ".join(detection.ocr_text for detection in ordered if detection.ocr_text)
        polygon = cls._bbox_to_polygon(bbox)

        return Detection(
            detect_type="text",
            confidence=confidence,
            geom_px={
                "type": "Polygon",
                "coordinates": [polygon + [polygon[0]]],
            },
            bbox_px=bbox,
            label="ocr_text",
            ocr_text=text,
        )

    @staticmethod
    def _union_bbox(bboxes: List[List[float]]) -> List[float]:
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    @staticmethod
    def _bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
        x, y, width, height = bbox
        right = x + width
        bottom = y + height
        return [
            [x, y],
            [right, y],
            [right, bottom],
            [x, bottom],
        ]

    @staticmethod
    def _horizontal_overlap_ratio(
        first_bbox: List[float],
        second_bbox: List[float],
    ) -> float:
        first_left = first_bbox[0]
        first_right = first_bbox[0] + first_bbox[2]
        second_left = second_bbox[0]
        second_right = second_bbox[0] + second_bbox[2]

        overlap = max(0.0, min(first_right, second_right) - max(first_left, second_left))
        smaller_width = min(first_bbox[2], second_bbox[2])
        if smaller_width <= 0:
            return 0.0
        return overlap / smaller_width
