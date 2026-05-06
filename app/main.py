"""
AI 도면 분석 서버 - 메인 진입점

이 서버는 Spring Boot 백엔드로부터 HTTP 요청을 받아
도면 이미지를 분석한 결과를 JSON으로 반환한다.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Indoor Navigation AI Service",
    description="실내 도면을 AI로 분석하는 서비스",
    version="0.1.0",
)


@app.get("/")
def root():
    """서버 살아있는지 확인용"""
    return {"message": "AI Service is running"}


@app.get("/health")
def health_check():
    """헬스체크 - 백엔드가 호출"""
    return {"status": "healthy", "service": "indoor-nav-ai"}