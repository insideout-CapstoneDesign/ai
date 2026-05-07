"""
도면 분석 API 라우터.

이 라우터는 오케스트레이터 역할을 한다:
1. 백엔드로부터 분석 요청 수신
2. 각 detector(text, object, poi, graph)를 순서대로 호출
3. 결과를 합쳐서 응답 반환

각 detector의 실제 구현은 app/services/ 안에 있으며, Detector 베이스 클래스를
상속받아 같은 인터페이스(detect 메서드)를 따른다.
"""

import time

from fastapi import APIRouter

from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.services.text_detector import TextDetector
from app.services.object_detector import ObjectDetector
from app.services.poi_detector import PoiDetector
from app.services.graph_detector import GraphDetector


router = APIRouter(
    prefix="/api/v1",
    tags=["analyze"],
)


# Detector 인스턴스 생성 (모듈 로드 시 한 번만)
# 추후 모델 로딩 무거워지면 FastAPI Dependency Injection으로 변경 가능
text_detector = TextDetector()
object_detector = ObjectDetector()
poi_detector = PoiDetector()
graph_detector = GraphDetector()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="도면 이미지 분석",
    description="""
    백엔드가 도면 이미지 URL을 보내면 AI가 분석하여 객체 감지 결과를 반환한다.
    
    내부적으로 4개의 detector를 순차 호출:
    1. TextDetector: OCR로 텍스트 추출
    2. ObjectDetector: YOLO로 벽/문/엘리베이터 감지
    3. PoiDetector: 텍스트 결과 활용해 POI 후보 식별
    4. GraphDetector: 오브젝트 결과로 노드/엣지 후보 생성
    
    현재는 모든 detector가 더미 데이터를 반환하는 스켈레톤 단계.
    """,
)
def analyze_floorplan(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    도면 분석 엔드포인트.
    
    각 detector를 호출하여 결과를 합쳐 반환한다. 의존성이 있는 detector는
    이전 결과를 인자로 받는다 (POI는 text 결과, Graph는 object 결과 활용).
    """
    start_time = time.time()

    # TODO: 실제로는 image_url에서 이미지 다운로드 후 image_path로 전달
    # 현재는 더미라 image_path를 그냥 image_url로 넘김
    image_path = request.image_url

    # 1) 텍스트 추출
    texts = text_detector.detect(image_path)

    # 2) 오브젝트 추출 (텍스트와 독립)
    objects = object_detector.detect(image_path)

    # 3) POI 추출 (텍스트 결과 활용)
    pois = poi_detector.detect(image_path, text_detections=texts)

    # 4) 노드·엣지 추출 (오브젝트 결과 활용)
    graph = graph_detector.detect(image_path, object_detections=objects)

    # 모든 결과 합치기
    all_detections = texts + objects + pois + graph

    elapsed_ms = int((time.time() - start_time) * 1000)

    return AnalyzeResponse(
        floorplan_id=request.floorplan_id,
        model_version=(request.options.model_version
                       if request.options else "v1.0"),
        processing_time_ms=elapsed_ms,
        detections=all_detections,
    )