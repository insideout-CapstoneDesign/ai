# Insideout AI Service

실내 도면 이미지를 AI로 분석하여 노드, 엣지, POI 후보를 추출하는 FastAPI 서버입니다.
백엔드(Spring Boot)로부터 분석 요청을 받아 결과를 JSON으로 반환합니다.



## 👥 팀원 소개 (Contributors)

> **Insideout 프로젝트를 이끈 양양양말을 소개합니다.**


| **차승은** | **이민지** | **김민준** | **김세현** |
| :---: | :---: | :---: | :---: |
| [<img src="https://github.com/user-attachments/assets/35081664-ee95-49bf-9bbf-0340df69f54b" height="180" width="130" style="border-radius: 8px;"><br/>](https://github.com/cktmddms) | [<img src="https://github.com/user-attachments/assets/8d75a543-b6ef-4a57-86c2-e06d93e9376d" height="180" width="130" style="border-radius: 8px;"><br/>](https://github.com/thisminji) | [<img src="https://github.com/user-attachments/assets/d6335e5f-31a8-4ab6-9432-1269227ae012" height="180" width="130" style="border-radius: 8px;"><br/>](https://github.com/minjune0) | [<img src="https://github.com/user-attachments/assets/40120ba5-e3c7-4048-9d54-cdfa837f7a6d" height="180" width="130" style="border-radius: 8px;"><br/>](https://github.com/sekong11) |
| 🔹 **Hybrid Navigation** <br> <sub>사용자 웹 - BE, FE</sub> | 🔹 **Auth, Search, Infra** <br> <sub>사용자 웹 - BE, FE</sub> | 🔹 **AI Map Builder** <br> <sub>관리자 웹 - AI, FE</sub> | 🔹 **Map Editor** <br> <sub>관리자 웹 - BE, FE</sub> |


## 🛠 기술 스택

### Core
<div>
  <img src="https://img.shields.io/badge/Python%203.11+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Uvicorn-646CFF?style=for-the-badge">
</div>

### AI / CV
<div>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/YOLO-111111?style=for-the-badge">
  <img src="https://img.shields.io/badge/EasyOCR-4B8BBE?style=for-the-badge">
</div>

### Validation
<div>
  <img src="https://img.shields.io/badge/Pydantic%20v2-E92063?style=for-the-badge">
</div>

## 📋 요구사항

- **Python 3.11 이상** (안정성 및 향후 호환성을 위해 3.11+ 권장)
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

### Windows

1. [python.org/downloads](https://www.python.org/downloads/)에서 **Python 3.11** Windows installer를 다운로드합니다.
2. 설치 첫 화면에서 **"Add python.exe to PATH"** 를 반드시 체크한 후 `Install Now`를 클릭합니다.
3. 설치 완료 후 CMD 또는 PowerShell에서 버전을 확인합니다:
```bat
   python --version
   # Python 3.11.x 가 떠야 정상
   ```
4. 가상환경 활성화 명령어는 **Windows에서 다릅니다** 2번 내용이 아닌 이 내용을 참고해주세요:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```

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

```text
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

```text
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

```text
type: 설명 (#이슈번호)

예시:
feat: 분석 API 스켈레톤 추가 (#4)
fix: model_version null 처리 (#4)
docs: README 작성 (#5)
```
