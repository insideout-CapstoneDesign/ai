"""
백엔드 ↔ AI 서버 간 통신 규격 정의.

Pydantic 모델은 다음 일을 한다:
1. 들어오는 JSON을 Python 객체로 자동 변환 (역직렬화)
2. 형식이 맞는지 검증 (예: floorplan_id가 진짜 UUID인지)
3. Swagger 문서 자동 생성
4. 응답을 JSON으로 자동 변환 (직렬화)

Spring Boot의 DTO + @Valid + Jackson을 한꺼번에 해주는 셈.
"""

from typing import Optional, List, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field


# ============================
# 요청 (백엔드 → AI 서버)
# ============================

class AnalyzeOptions(BaseModel):
    """AI 분석 시 사용할 옵션 (선택사항)."""

    model_version: str = Field(
        default="v1.0",
        min_length=1,
        description="사용할 AI 모델 버전",
        examples=["v1.0"],
    )


class AnalyzeRequest(BaseModel):
    """백엔드가 AI 서버에 도면 분석을 요청할 때의 데이터 형식."""

    floorplan_id: UUID = Field(
        ...,
        description="DB의 floorplan 테이블 PK (분석 결과 연결용)",
        examples=["3f7d4b8c-1234-5678-9abc-def012345678"],
    )
    image_url: str = Field(
        ...,
        description="분석할 도면 이미지 URL (S3/MinIO)",
        examples=["https://s3.../building1/floor2.png"],
    )
    options: Optional[AnalyzeOptions] = Field(
        default=None,
        description="분석 옵션 (선택)",
    )


# ============================
# 응답 (AI 서버 → 백엔드)
# ============================

class Detection(BaseModel):
    """AI가 감지한 객체 하나.
    
    detect_type을 Literal로 제한해서 정의된 값 외엔 422 에러.
    """

    detect_type: Literal[
        # 실내
        "wall", "door", "corridor", "room",
        "elevator", "stair", "escalator", "restroom_sign",
        # 외부 (단지)
        "building", "gate", "road", "sidewalk",
        "crosswalk", "parking", "landmark",
        # POI 아이콘
        "accessible_restroom", "aed", "atm", "cafe",
        "clothing_alteration", "family_restroom", "infodesk",
        "phone_charging", "restroom_female", "restroom_male",
        "shoe_repair", "storage_locker", "subway_station",
        "water_fountain",
        # 공통
        "text", "poi_candidate", "node_candidate", "edge_candidate",
    ] = Field(
        ...,
        description="감지된 객체 타입",
        examples=["wall"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="신뢰도 (0~1)",
        examples=[0.92],
    )
    geom_px: Dict[str, Any] = Field(
        ...,
        description="감지 영역의 도면 픽셀 좌표 (GeoJSON 형식)",
        examples=[{
            "type": "LineString",
            "coordinates": [[100, 200], [500, 200]]
        }],
    )
    bbox_px: Optional[List[float]] = Field(
        default=None,
        description="바운딩 박스 [x, y, width, height]",
        examples=[[100, 200, 400, 50]],
    )
    label: Optional[str] = Field(
        default=None,
        description="모델이 출력한 원본 라벨",
    )
    ocr_text: Optional[str] = Field(
        default=None,
        description="OCR 결과 텍스트 (text 타입일 때)",
        examples=["202호"],
    )


class AnalyzeResponse(BaseModel):
    """AI 서버가 백엔드에게 돌려주는 분석 결과."""

    floorplan_id: UUID = Field(
        ...,
        description="요청에 포함되었던 floorplan_id (그대로 반환)",
    )
    model_version: str = Field(
        ...,
        description="실제 사용된 모델 버전",
    )
    processing_time_ms: int = Field(
        ...,
        description="분석 소요 시간 (밀리초)",
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="감지된 객체 목록",
    )
