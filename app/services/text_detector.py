"""
텍스트(OCR) 추출 Detector.

도면 이미지에서 텍스트(상점명, 호실 번호 등)를 OCR로 추출한다.

현재는 더미 데이터를 반환하는 스켈레톤. 실제 OCR 로직(EasyOCR, Tesseract)은
별도 이슈에서 구현 예정.
"""

from typing import List

from app.schemas.analyze import Detection
from app.services.base import Detector


class TextDetector(Detector):
    """OCR을 사용해 도면의 텍스트를 추출한다."""

    name = "text"
    version = "v0.1-dummy"

    def detect(self, image_path: str) -> List[Detection]:
        """
        이미지에서 텍스트를 추출.
        """
        # TODO: EasyOCR/Tesseract로 실제 OCR 수행
        return [
            Detection(
                detect_type="text",
                confidence=0.95,
                geom_px={
                    "type": "Point",
                    "coordinates": [350, 250],
                },
                bbox_px=[340, 240, 80, 25],
                ocr_text="202호",
                label="room_number",
            ),
            Detection(
                detect_type="text",
                confidence=0.92,
                geom_px={
                    "type": "Point",
                    "coordinates": [800, 400],
                },
                bbox_px=[790, 390, 100, 25],
                ocr_text="화장실",
                label="facility_label",
            ),
        ]