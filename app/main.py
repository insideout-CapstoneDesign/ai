"""
AI 도면 분석 서버 - 메인 진입점
"""

from fastapi import FastAPI

from app.routers import analyze       
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="실내 도면을 AI로 분석하는 서비스",
    version="0.1.0",
)

app.include_router(analyze.router)   


@app.get("/")
def root():
    return {"message": "AI Service is running"}


@app.get("/health")
def health_check():
    """헬스체크 - 환경 정보도 함께 반환"""
    return {
        "status": "healthy",
        "service": "indoor-nav-ai",
        "version": settings.app_version,
        "environment": settings.environment,
    }