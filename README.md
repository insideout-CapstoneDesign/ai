# Insideout AI Service

실내 도면 이미지를 AI로 분석하여 노드, 엣지, POI 후보를 추출하는 FastAPI 서버입니다.
백엔드(Spring Boot)로부터 분석 요청을 받아 결과를 JSON으로 반환합니다.

## 🛠 기술 스택

- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Server**: Uvicorn
- **AI/CV**: OpenCV, YOLO, EasyOCR (예정)
- **Validation**: Pydantic v2

## 📋 요구사항

- **Python 3.11 이상** (PEP 604 union 문법 사용으로 3.11+ 필수)
- macOS / Linux (Windows는 WSL 권장)
- 4GB 이상 RAM (AI 모델 로딩 시)

## 🚀 빠른 시작

### 1. 사전 준비

Python 3.11이 설치되어 있는지 확인:

```bash
python3.11 --version
# Python 3.11.x 가 떠야 정상
```

없다면 Homebrew(MAC)로 설치:

```bash
brew install python@3.11
```
윈도우는 찾아봐주세요 ..

### 2. 프로젝트 클론 및 가상환경 설정

```bash
git clone https://github.com/insideout-CapstoneDesign/ai.git
cd ai

# 가상환경 생성 (반드시 python3.11 명시)
python3.11 -m venv .venv

# 활성화
source .venv/bin/activate

# 프롬프트에 (.venv)가 보이면 OK
```

### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

### 5. 동작 확인

| URL | 설명 |
|---|---|
| `http://localhost:8000/` | 서버 상태 메시지 |
| `http://localhost:8000/health` | 헬스체크 |
| `http://localhost:8000/docs` | **Swagger UI (API 문서)** |

## 📁 프로젝트 구조

```
ai/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 진입점
│   ├── routers/
│   │   ├── __init__.py
│   │   └── analyze.py             # /api/v1/analyze 엔드포인트
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── analyze.py             # 요청/응답 Pydantic 모델
│   └── services/
│       ├── __init__.py
│       ├── base.py                # Detector 추상 베이스 클래스
│       ├── text_detector.py       # OCR 추출
│       ├── object_detector.py     # 벽/문/엘리베이터 등 감지
│       ├── poi_detector.py        # POI 후보 추출
│       └── graph_detector.py      # 노드/엣지 후보 추출
├── .gitignore
├── README.md
└── requirements.txt
```

## 🔌 API

전체 API 문서는 서버 실행 후 [Swagger UI](http://localhost:8000/docs)에서 확인 가능합니다.

### 주요 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 서버 상태 확인 |
| GET | `/health` | 헬스체크 |
| POST | `/api/v1/analyze` | 도면 이미지 분석 |

## 🏗 아키텍처

```
[Spring Boot 백엔드] ──HTTP POST──▶ [FastAPI AI 서버]
                                          │
                                          ├─ TextDetector  (OCR)
                                          ├─ ObjectDetector (YOLO/OpenCV)
                                          ├─ PoiDetector   (POI 추출)
                                          └─ GraphDetector (노드/엣지)
                                          │
                                          ▼ JSON 응답
                                    [백엔드가 DB에 저장]
```

각 detector는 `Detector` 추상 베이스 클래스를 상속받아 동일한 인터페이스를 따릅니다.

## 👥 개발 가이드

### 가상환경 종료

```bash
deactivate
```

### 새 의존성 추가 시

```bash
pip install <패키지명>
pip freeze > requirements.txt
```

### 브랜치 전략

- `main`: 배포 가능 상태 (직접 push 금지)
- `dev`: 개발 통합 브랜치
- `feat/이슈번호-설명`: 기능 브랜치

### 커밋 컨벤션

```
type: 설명 (#이슈번호)

예시:
feat: 분석 API 스켈레톤 추가 (#4)
fix: model_version null 처리 (#4)
docs: README 작성 (#5)
```
