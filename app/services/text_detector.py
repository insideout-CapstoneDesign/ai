"""OCR text detector for floorplan images."""

import logging
import os
from pathlib import Path
from typing import Any, List

from app.core.config import settings
from app.schemas.analyze import Detection
from app.services.base import Detector

logger = logging.getLogger(__name__)


class TextDetector(Detector):
    """Extract floorplan text with the configured OCR engine."""

    name = "text"

    def __init__(self, engine: str | None = None) -> None:
        self.engine = engine or settings.ocr_engine
        if self.engine not in {"easyocr", "paddleocr"}:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
        self.version = (
            "paddleocr-3.6.0"
            if self.engine == "paddleocr"
            else "easyocr-1.7.2"
        )
        self._reader: Any | None = None

    def detect(
        self,
        image_path: str,
        object_detections: List[Detection] | None = None,
    ) -> List[Detection]:
        """Extract text boxes and normalize them to the Detection schema."""
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if self.engine == "paddleocr":
            detections = self._detect_with_paddleocr(image_path)
        else:
            detections = self._detect_with_easyocr(image_path)

        detections = self._exclude_icon_overlaps(
            detections,
            object_detections or [],
        )
        detections = self._merge_stacked_texts(detections)
        logger.info(
            "Detected %d text regions from %s with %s",
            len(detections),
            image_path,
            self.engine,
        )
        return detections

    def _detect_with_easyocr(self, image_path: str) -> List[Detection]:
        raw_results = self._get_easyocr_reader().readtext(image_path)
        detections: List[Detection] = []
        for raw_bbox, raw_text, raw_confidence in raw_results:
            detection = self._to_detection(raw_bbox, raw_text, raw_confidence)
            if detection is not None:
                detections.append(detection)
        return detections

    def _detect_with_paddleocr(self, image_path: str) -> List[Detection]:
        detections: List[Detection] = []
        for result in self._get_paddleocr_reader().predict(image_path):
            polygons = result.get("rec_polys", [])
            texts = result.get("rec_texts", [])
            scores = result.get("rec_scores", [])
            for polygon, text, score in zip(polygons, texts, scores):
                detection = self._to_detection(polygon, text, score)
                if detection is not None:
                    detections.append(detection)
        return detections

    def _to_detection(
        self,
        raw_bbox: Any,
        raw_text: Any,
        raw_confidence: Any,
    ) -> Detection | None:
        text = str(raw_text).strip()
        confidence = self._clamp_confidence(raw_confidence)
        if not text or confidence < settings.ocr_confidence_threshold:
            return None

        polygon = self._normalize_polygon(raw_bbox)
        bbox = self._polygon_to_bbox(polygon)
        if self._is_icon_like_text_artifact(text, bbox):
            return None

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

    def _get_easyocr_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "EasyOCR is not installed. Run 'pip install -r requirements.txt'."
                ) from exc

            self._reader = easyocr.Reader(
                self._easyocr_languages(),
                gpu=settings.ocr_gpu,
                model_storage_directory=settings.ocr_model_storage_dir,
                download_enabled=settings.ocr_download_enabled,
                verbose=False,
            )
        return self._reader

    def _get_paddleocr_reader(self) -> Any:
        if self._reader is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            try:
                # On Windows, loading torch first avoids a Paddle/PyTorch DLL conflict.
                import torch  # noqa: F401
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR is not installed. Run 'pip install -r requirements.txt'."
                ) from exc

            self._reader = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=settings.paddleocr_enable_mkldnn,
            )
        return self._reader

    @staticmethod
    def _easyocr_languages() -> List[str]:
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

    @staticmethod
    def _is_icon_like_text_artifact(text: str, bbox: List[float]) -> bool:
        """Reject short OCR results that are likely symbols inside square icons."""
        _, _, width, height = bbox
        compact_text = "".join(text.split())
        is_short_ascii = compact_text.isascii() and len(compact_text) <= 2
        is_single_non_ascii = not compact_text.isascii() and len(compact_text) == 1
        if (
            not (is_short_ascii or is_single_non_ascii)
            or min(width, height) < 24
        ):
            return False

        aspect_ratio = width / height if height else 0.0
        return 0.65 <= aspect_ratio <= 1.35

    @classmethod
    def _exclude_icon_overlaps(
        cls,
        text_detections: List[Detection],
        object_detections: List[Detection],
    ) -> List[Detection]:
        """Remove OCR regions located inside detected POI icon boxes."""
        icon_boxes = [
            detection.bbox_px
            for detection in object_detections
            if detection.bbox_px is not None
        ]
        if not icon_boxes:
            return text_detections

        return [
            detection
            for detection in text_detections
            if detection.bbox_px is None
            or not any(
                cls._bbox_overlap_ratio(detection.bbox_px, icon_box) >= 0.5
                for icon_box in icon_boxes
            )
        ]

    @staticmethod
    def _bbox_overlap_ratio(first_bbox: List[float], second_bbox: List[float]) -> float:
        """Return intersection area as a ratio of the first box area."""
        first_x, first_y, first_width, first_height = first_bbox
        second_x, second_y, second_width, second_height = second_bbox
        intersection_width = max(
            0.0,
            min(first_x + first_width, second_x + second_width)
            - max(first_x, second_x),
        )
        intersection_height = max(
            0.0,
            min(first_y + first_height, second_y + second_height)
            - max(first_y, second_y),
        )
        first_area = first_width * first_height
        if first_area <= 0:
            return 0.0
        return intersection_width * intersection_height / first_area

    @classmethod
    def _merge_stacked_texts(cls, detections: List[Detection]) -> List[Detection]:
        """Merge tightly stacked OCR lines into one text detection."""
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
