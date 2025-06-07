
# 📋 URS & SRS 문서 작성 계획

프로젝트명: **Semiconductor AOI Inspector**  
작성일: 2025-06-08  
작성자: 김응수

---

## ✅ 전체 URS/SRS 작성 대상 목록

SDD 문서 기반으로 파악한 모듈 및 구성요소에 대해 아래와 같이 URS/SRS 문서 작성이 필요합니다.

---

### 🧩 1단계 (MVP 핵심 모듈: 반드시 작성)

| 모듈명            | 설명                          | URS  | SRS  |
|-------------------|-------------------------------|------|------|
| `data_loader.py`  | 이미지 폴더 내 이미지 로딩   | ✅ 작성 중 | ✅ 작성 중 |
| `preprocessor.py` | 이미지 정규화 및 전처리       | ⬜ 미작성 | ⬜ 미작성 |
| `inference.py`    | 모델 로딩 및 추론             | ⬜ 미작성 | ⬜ 미작성 |
| `postprocessor.py`| 추론 결과 요약/필터링         | ⬜ 미작성 | ⬜ 미작성 |
| `visualizer.py`   | 결과 시각화 및 저장           | ⬜ 미작성 | ⬜ 미작성 |
| `main.py`         | 전체 실행 흐름 통합           | ⬜ 미작성 | ⬜ 미작성 |

---

### 🧪 2단계 (학습/성능 평가/자동화 확장용)

| 모듈명           | 설명                          | URS  | SRS  |
|------------------|-------------------------------|------|------|
| `train.py`       | fine-tuning 학습 파이프라인   | ⬜ 미작성 | ⬜ 미작성 |
| `eval.py`        | F1-score 등 성능 평가         | ⬜ 미작성 | ⬜ 미작성 |
| `config.yaml`    | 설정 기반 구조화              | ⬜ 미작성 | ⬜ 미작성 |

---

### 🧠 3단계 (선택적 또는 장기 고려 대상)

| 항목              | 설명                                | URS  | SRS  |
|-------------------|-------------------------------------|------|------|
| API 서버          | FastAPI 기반 API 통합               | ⬜ 미작성 | ⬜ 미작성 |
| ONNX/TensorRT     | 모델 최적화 및 배포용               | ⬜ 미작성 | ⬜ 미작성 |
| MLOps 지원        | 자동화 학습/추론/로깅 파이프라인    | ⬜ 미작성 | ⬜ 미작성 |

---

## 📌 작성 추천 순서

1. `data_loader.py` 확정
2. 이후 `preprocessor.py` → `inference.py` → `postprocessor.py` 순 진행
3. 공통된 입력/출력/구조는 템플릿화 가능

---

## 📂 예시 디렉토리 구조

```
project_root/
├── main.py
├── modules/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── inference.py
│   ├── postprocessor.py
│   └── visualizer.py
├── models/
│   └── yolov5s.pt
├── config.yaml
├── requirements.txt
└── README.md
```
