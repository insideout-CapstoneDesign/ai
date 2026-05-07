"""
애플리케이션 환경 설정.

이 모듈은 .env 파일과 환경 변수에서 설정값을 읽어 타입 안전한 Settings 객체로 제공한다.

사용 예:
    from app.core.config import settings
    print(settings.app_name)
    print(settings.ai_service_port)

Spring Boot의 @ConfigurationProperties 와 같은 역할.
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정.
    
    Pydantic이 자동으로:
    1. .env 파일에서 값 로드
    2. 환경 변수에서 값 로드 (.env 보다 우선)
    3. 타입 검증 (str → int 등 자동 변환)
    4. 누락된 필수 값 발견 시 시작 거부
    """

    # SettingsConfigDict: 어디서 환경변수 읽을지 설정
    model_config = SettingsConfigDict(
        env_file=".env",                    # .env 파일에서 읽기
        env_file_encoding="utf-8",
        case_sensitive=False,               # 대소문자 무시 (S3_KEY = s3_key)
        extra="ignore",                     # 정의 안 된 변수는 무시
    )

    # ============================
    # 애플리케이션 기본 정보
    # ============================
    app_name: str = Field(
        default="InSideOut AI Service",
        description="애플리케이션 이름",
    )
    app_version: str = Field(
        default="0.1.0",
        description="애플리케이션 버전",
    )
    environment: Literal["local", "dev", "prod"] = Field(
        default="local",
        description="실행 환경",
    )

    # ============================
    # 서버 설정
    # ============================
    ai_service_host: str = Field(
        default="0.0.0.0",
        description="서버 바인딩 호스트",
    )
    ai_service_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="서버 포트",
    )

    # ============================
    # 로깅
    # ============================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="로그 레벨",
    )

    # ============================
    # 외부 서비스 (백엔드)
    # ============================
    backend_url: str = Field(
        default="http://localhost:8080",
        description="Spring Boot 백엔드 URL (콜백/통합 시 사용)",
    )

    # ============================
    # S3 / MinIO (도면 이미지 저장소) — 실제 AI 구현 시 사용
    # ============================
    s3_endpoint: Optional[str] = Field(
        default=None,
        description="S3/MinIO 엔드포인트 URL (없으면 AWS S3)",
    )
    s3_region: str = Field(
        default="ap-northeast-2",
        description="S3 리전",
    )
    s3_access_key: Optional[str] = Field(
        default=None,
        description="S3 액세스 키",
    )
    s3_secret_key: Optional[str] = Field(
        default=None,
        description="S3 시크릿 키",
    )
    s3_bucket: str = Field(
        default="indoor-nav-floorplans",
        description="도면 이미지가 저장된 버킷 이름",
    )

    # ============================
    # AI 모델 경로 — 실제 AI 구현 시 사용
    # ============================
    yolo_model_path: Optional[str] = Field(
        default=None,
        description="YOLO 모델 가중치 파일 경로",
    )
    ocr_language: str = Field(
        default="ko",
        description="OCR 인식 언어 (ko/en)",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Settings 인스턴스를 반환하는 함수.
    
    @lru_cache 덕분에 한 번 만들어진 Settings 객체가 재사용됨 (Singleton 패턴).
    매번 .env 파일을 다시 읽지 않아 효율적.
    """
    return Settings()


# 모듈 임포트 시 즉시 인스턴스 생성 (어디서든 settings 참조 가능)
settings = get_settings()