"""
AI 도면 분석 서버 - 메인 진입점
"""

from fastapi import FastAPI

from app.routers import analyze       # ← 1. import

app = FastAPI(
    title="Indoor Navigation AI Service",
    description="실내 도면을 AI로 분석하는 서비스",
    version="0.1.0",
)

app.include_router(analyze.router)    # ← 2. 등록


@app.get("/")
def root():
    return {"message": "AI Service is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "indoor-nav-ai"}