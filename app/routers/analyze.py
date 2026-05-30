"""
도면 분석 API 라우터.

이 라우터는 오케스트레이터 역할을 한다:
1. 백엔드로부터 분석 요청 수신
2. 각 detector(text, object, poi, graph)를 순서대로 호출
3. 결과를 합쳐서 응답 반환

각 detector의 실제 구현은 app/services/ 안에 있으며, Detector 베이스 클래스를
상속받아 같은 인터페이스(detect 메서드)를 따른다.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.core.image_loader import download_image, cleanup_image
from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
)
from app.services.text_detector import TextDetector
from app.services.object_detector import ObjectDetector
from app.services.poi_detector import PoiDetector
from app.services.structure_detector import StructureDetector
from app.services.graph_detector import GraphDetector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["analyze"],
)


# Detector 인스턴스 생성 (모듈 로드 시 한 번만)
# 추후 모델 로딩 무거워지면 FastAPI Dependency Injection으로 변경 가능
text_detector = TextDetector()
object_detector = ObjectDetector()
poi_detector = PoiDetector()
structure_detector = StructureDetector()
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

    흐름:
    1. image_url에서 이미지 다운로드 (S3 또는 HTTP)
    2. 4개 detector 순차 호출
    3. 결과 합쳐 반환
    4. 임시 파일 정리 (성공/실패 무관)
    """
    start_time = time.time()
    local_path = None                        

    try:                                       
        # ✏️ 변경: 이전엔 image_path = request.image_url 였음
        # 이제는 진짜 다운로드
        try:
            local_path = download_image(request.image_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except IOError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Image fetch failed: {e}",
            )

        # 1) 텍스트 추출
        texts = text_detector.detect(local_path)     # ✏️ image_path → local_path

        # 2) 오브젝트 추출
        objects = object_detector.detect(local_path) 

        # 3) POI 추출 (텍스트 결과 활용)
        pois = poi_detector.detect(
            local_path,
            text_detections=texts,
            object_detections=objects,
        )

        # 4) 노드·엣지 추출 (오브젝트 결과 활용)
        structures = structure_detector.detect(
            local_path,
            text_detections=texts,
            object_detections=objects,
        )

        graph = graph_detector.detect(
            local_path,
            object_detections=objects,
            structure_detections=structures,
            poi_detections=pois,
        )

        all_detections = texts + objects + pois + structures + graph

        elapsed_ms = int((time.time() - start_time) * 1000)

        return AnalyzeResponse(
            floorplan_id=request.floorplan_id,
            model_version=(request.options.model_version
                           if request.options else "v1.0"),
            processing_time_ms=elapsed_ms,
            detections=all_detections,
        )

    finally:                                        
        # 임시 파일 정리 (성공/실패 무관)
        if local_path:
            cleanup_image(local_path)   
