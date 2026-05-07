"""
오브젝트 추출 Detector.

도면 이미지에서 물리적 구조 객체(벽, 문, 엘리베이터, 계단 등)를 감지한다.

현재는 더미 데이터. 실제 구현 시 YOLO 또는 OpenCV 사용 예정.
"""

from typing import List

from app.schemas.analyze import Detection
from app.services.base import Detector


class ObjectDetector(Detector):
    """YOLO/OpenCV로 도면의 물리적 객체를 감지한다."""

    name = "object"
    version = "v0.1-dummy"

    def detect(self, image_path: str) -> List[Detection]:
        """
        이미지에서 벽, 문, 엘리베이터 등을 감지.
        """
        # TODO: YOLO로 엘리베이터/계단/문 등 객체 감지
        # TODO: OpenCV Hough Transform으로 벽 라인 추출
        return [
            Detection(
                detect_type="wall",
                confidence=0.92,
                geom_px={
                    "type": "LineString",
                    "coordinates": [[100, 200], [500, 200]],
                },
                bbox_px=[100, 195, 400, 10],
                label="wall_horizontal",
            ),
            Detection(
                detect_type="door",
                confidence=0.88,
                geom_px={
                    "type": "LineString",
                    "coordinates": [[300, 200], [320, 200]],
                },
                bbox_px=[300, 195, 20, 10],
                label="door_open",
            ),
            Detection(
                detect_type="elevator",
                confidence=0.94,
                geom_px={
                    "type": "Polygon",
                    "coordinates": [[[1200, 1500], [1250, 1500],
                                     [1250, 1560], [1200, 1560], [1200, 1500]]],
                },
                bbox_px=[1200, 1500, 50, 60],
                label="elevator",
            ),
        ]