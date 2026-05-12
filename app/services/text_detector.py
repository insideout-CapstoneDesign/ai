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
