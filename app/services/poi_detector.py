"""
POI(관심 지점) 추출 Detector.

도면에서 매장, 시설, 강의실 등 사용자가 검색·이동하는 의미 있는 장소를 식별한다.

현재는 더미 데이터. 실제 구현 시 텍스트 결과를 활용하여 POI 후보를 생성.
"""

from typing import List

from app.schemas.analyze import Detection
from app.services.base import Detector


class PoiDetector(Detector):
    """도면에서 POI 후보를 식별한다."""

    name = "poi"
    version = "v0.1-dummy"

    def detect(
        self,
        image_path: str,
        text_detections: List[Detection] | None = None,
    ) -> List[Detection]:
        """
        이미지에서 POI 후보를 추출.
        
        Args:
            image_path: 이미지 경로
            text_detections: 텍스트 추출 결과 (POI 이름 후보로 활용)
        PoiDetector.detect()가 text_detections 파라미터를 추가로 받는 이유: POI는 텍스트(매장명, 호실번호) 위치를 활용해야 효율적이라서. 
        베이스 클래스의 시그니처와 약간 다른데, Python은 이런 유연한 확장을 허용
        """
        # TODO: text_detections를 보고 POI 후보 영역 추정
        return [
            Detection(
                detect_type="poi_candidate",
                confidence=0.80,
                geom_px={
                    "type": "Point",
                    "coordinates": [350, 280],
                },
                ocr_text="202호",
                label="classroom_candidate",
            ),
        ]