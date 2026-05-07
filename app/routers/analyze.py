"""
도면 분석 API 라우터.

Spring Boot의 @RestController + @RequestMapping과 같은 역할.
관련 엔드포인트들을 한 파일에 묶어서 main.py가 앱에 등록한다.
"""

import time
from uuid import UUID
from fastapi import APIRouter, HTTPException

from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    Detection,
)


# APIRouter는 "엔드포인트 모음".
# prefix를 주면 모든 경로 앞에 자동으로 붙음.
# (예: /analyze가 자동으로 /api/v1/analyze가 됨)
router = APIRouter(
    prefix="/api/v1",
    tags=["analyze"],          # Swagger에서 그룹핑 라벨
)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,    # 응답 형식 명시 (검증 + Swagger)
    summary="도면 이미지 분석",
    description="""
    백엔드가 도면 이미지 URL을 보내면 AI가 분석하여 객체 감지 결과를 반환한다.
    
    ⚠️ 현재는 더미 응답을 반환하는 스켈레톤 단계.
    실제 OpenCV/YOLO/OCR 처리는 다음 이슈에서 추가 예정.
    """,
)
def analyze_floorplan(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    도면 분석 엔드포인트.
    
    @router.post("/analyze")가 만나게 되는 흐름:
    1. FastAPI가 들어온 JSON을 AnalyzeRequest로 자동 변환·검증
    2. 검증 통과하면 이 함수 실행
    3. 반환값을 JSON으로 자동 직렬화하여 응답
    """
    start_time = time.time()
    
    # ⚠️ 더미 응답 - 실제 AI 분석은 다음 이슈에서 구현
    dummy_detections = [
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
        ),
        Detection(
            detect_type="text",
            confidence=0.95,
            geom_px={
                "type": "Point",
                "coordinates": [350, 250],
            },
            ocr_text="202호",
            label="room_number",
        ),
    ]
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    return AnalyzeResponse(
        floorplan_id=request.floorplan_id,
        model_version=(request.options.model_version 
                       if request.options else "v1.0"),
        processing_time_ms=elapsed_ms,
        detections=dummy_detections,
    )