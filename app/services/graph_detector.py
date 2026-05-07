"""
노드·엣지 추출 Detector.

오브젝트(벽, 문) 추출 결과를 기반으로 길찾기에 필요한 노드(교차점, 출입구)와
엣지(이동 가능한 경로)를 추출한다.

현재는 더미 데이터. 실제 구현 시 NetworkX 등 그래프 라이브러리 활용 예정.
"""

from typing import List

from app.schemas.analyze import Detection
from app.services.base import Detector


class GraphDetector(Detector):
    """오브젝트 결과로부터 위상 그래프(노드/엣지) 후보를 추출한다."""

    name = "graph"
    version = "v0.1-dummy"

    def detect(
        self,
        image_path: str,
        object_detections: List[Detection] | None = None,
    ) -> List[Detection]:
        """
        오브젝트 결과를 기반으로 노드·엣지 후보 생성.
        
        Args:
            image_path: 이미지 경로
            object_detections: 오브젝트 추출 결과 (벽, 문 위치 활용)
        
        """
        # TODO: object_detections의 벽/문 위치 분석하여 통로(엣지) 추출
        # TODO: 통로 교차점에 노드 후보 배치
        return [
            Detection(
                detect_type="node_candidate",
                confidence=0.82,
                geom_px={
                    "type": "Point",
                    "coordinates": [1000, 1700],
                },
                label="corridor_intersection",
            ),
            Detection(
                detect_type="edge_candidate",
                confidence=0.78,
                geom_px={
                    "type": "LineString",
                    "coordinates": [[1000, 1700], [1500, 1700]],
                },
                label="walkway",
            ),
        ]