
# 🔧 소프트웨어 설계 문서 (SDD)
**프로젝트명:** Semiconductor AOI Inspector (가칭)  
**작성일자:** 2025.06.08  
**작성자:** 김응수  
**문서 버전:** v1.0

---

## 1. 프로젝트 개요

| 항목 | 설명 |
|------|------|
| 목적 | 반도체 제조 공정에서 이물 및 결함을 자동으로 검출하기 위한 AI 기반의 검사 도구 개발 |
| 개발 플랫폼 | Python 3.x (CLI 환경) |
| 활용 분야 | 웨이퍼 표면 검사, bump/ball 결함 탐지, 이물 식별 등 |
| 핵심 기술 | 비지도 이상 탐지, 객체 검출, 전처리 및 후처리 모듈화 |
| 개발 도구 | GitHub Copilot(Codex), PyTorch, OpenCV, Anomalib, YOLOv5/YOLOv8, Albumentations 등 |
| 초기 대상 사용자 | 개발자 본인 (사이드 프로젝트 용도) |
| 배포 방식 | CLI 프로그램 (GUI 없음), 모듈형 구조 |
| 성능 목표 | 정밀도 90% 이상, 추론시간 500ms 이하 (장기) |

---

## 2. 개발 목표 (단계별)

| 단계 | 기간 | 목표 |
|------|------|------|
| 단기 | ~2개월 | 모듈 구조 구축, 사전 학습 모델 기반 프로토타입 완성 (MVP), 샘플 데이터 기반 실행 |
| 중기 | 3~6개월 | 자체 데이터 확보 및 증강, 성능 개선, 모델 튜닝, 평가 체계 구축 |
| 장기 | 6개월~1년 이상 | 자체 학습 가능 구조 도입, 제품화 고려, 타 공정/도메인 확장 설계 |

---

## 3. 전체 아키텍처

```
[main.py]
   |
   |-- data_loader.py       ← 이미지 불러오기
   |-- preprocessor.py      ← 이미지 전처리
   |-- inference.py         ← 모델 로딩 및 추론
   |-- postprocessor.py     ← 결과 후처리
   |-- visualizer.py        ← 시각화 결과 출력 (선택적)
   |-- config.yaml          ← 설정 파일
```

---

## 4. 모듈 설계

### 4.1. 데이터 로딩 모듈 (data_loader.py)
```python
def load_images_from_folder(folder_path: str) -> List[np.ndarray]
```

### 4.2. 전처리 모듈 (preprocessor.py)
```python
def preprocess_image(img: np.ndarray) -> np.ndarray
```

### 4.3. 모델 추론 모듈 (inference.py)
```python
def load_model(model_name: str): ...
def run_inference(img: np.ndarray): ...
```

### 4.4. 후처리 모듈 (postprocessor.py)
```python
def summarize_detection(results) -> str
```

### 4.5. 결과 출력/시각화 (visualizer.py)
```python
def save_result_image(img: np.ndarray, detections: List, path: str)
```

---

## 5. 데이터 전략

| 항목 | 설명 |
|------|------|
| 공개 데이터셋 | MVTec AD, PCB defect data (Kaggle) |
| 증강 도구 | Albumentations (기하, 밝기, 노이즈 등) |
| 자체 수집 | 공정 이미지, 현미경 촬영, 이물 합성 |
| 레이블링 | LabelMe, CVAT + 수동/반자동 방식 |
| 이상 탐지용 학습 | 정상 데이터 기반 (PatchCore 등) |

---

## 6. 학습 및 추론 전략

### 초기 (사전 학습 모델 사용)
- YOLOv5/YOLOv8 (객체 검출)
- Anomalib (PatchCore, Padim 등 이상 탐지)

### 중기 (미세조정 및 커스텀 학습)
- 자체 수집 데이터 기반 fine-tuning
- 분할 학습
- CLI 기반 학습 툴 분리 예정

---

## 7. 성능 평가 및 목표

| 항목 | 목표 기준 |
|------|------------|
| 정확도 | 이상/결함 탐지 시 F1-score > 90% |
| 속도 | 추론시간 500ms/이미지 이하 (단건) |
| 견고성 | 조명 변화, noise, orientation 변화 견딤 |
| 평가 도구 | confusion matrix, heatmap 시각화, 로그 저장 |

---

## 8. 유지보수 및 확장성

- 설정 파일 기반 구조 (config.yaml)
- 모듈 간 결합 최소화
- 함수 단위 단위 테스트 가능 구조
- API 확장 및 배포 용이

---

## 9. 기술 스택 요약

| 구분 | 기술 |
|------|------|
| 주요 언어 | Python 3.10 이상 |
| 모델 프레임워크 | PyTorch, Ultralytics YOLO, Anomalib |
| 이미지 처리 | OpenCV, NumPy |
| 데이터 증강 | Albumentations |
| 라벨링 도구 | LabelMe, CVAT |

---

## 10. 향후 고려 사항

- train.py: 사용자 커스텀 학습 모듈
- eval.py: 성능 평가 모듈
- Flask/FastAPI 백엔드 통합
- ONNX 및 TensorRT 변환
- 성능 최적화 및 병렬화 구조 적용

---

## 부록

📁 프로젝트 디렉토리 구조 예시:
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
