
# 📄 URS & SRS 상세 문서 v1.0
프로젝트명: Semiconductor AOI Inspector  
작성일: 2025-06-08  
작성자: 김응수

---

## ✅ 1. data_loader.py

### URS
- 사용자는 폴더 경로를 입력하여 내부의 모든 이미지를 자동으로 불러오고자 한다.
- 잘못된 파일(비이미지, 손상 이미지 등)은 자동으로 건너뛰어야 하며 사용자에게 경고를 표시한다.
- 불러온 이미지들의 순서는 추후 라벨이나 추론 결과와 매칭되도록 유지되어야 한다.

### SRS
- 입력: `folder_path (str)`
- 출력: `List[np.ndarray]`
- 처리:
  - `os.listdir()` + 확장자 필터링
  - OpenCV `cv2.imread`로 이미지 로딩
  - 로딩 실패 시 로그 출력
- 지원 포맷: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- 예외 처리: 폴더 없음, 비정상 파일, 빈 폴더

---

## ✅ 2. preprocessor.py

### URS
- 사용자는 다양한 크기/색상/조도 조건을 가진 이미지를 정규화된 형태로 전처리하고자 한다.
- 모든 이미지는 grayscale로 변환되고, 동일한 크기로 resize되어야 하며, 정규화 처리가 포함되어야 한다.

### SRS
- 입력: `img (np.ndarray)`
- 출력: `np.ndarray`
- 처리:
  - grayscale 변환 (`cv2.cvtColor`)
  - resize (640x640)
  - 정규화 (0~1 or mean/std)
- 선택 옵션: CLAHE, histogram equalization

---

## ✅ 3. inference.py

### URS
- 사용자는 사전 학습된 모델을 로딩하여 이미지에 대한 이상 탐지 또는 객체 검출을 수행하고자 한다.
- 다양한 모델 유형(PatchCore, YOLO 등)에 유연하게 대응해야 한다.

### SRS
- 함수:
  - `load_model(model_name: str) -> Any`
  - `run_inference(img: np.ndarray) -> List[Dict] or np.ndarray`
- 처리:
  - torch 또는 Anomalib 모델 로딩
  - GPU/CPU 자동 선택
  - 결과: bbox, score, anomaly map 등

---

## ✅ 4. postprocessor.py

### URS
- 사용자는 추론 결과로부터 유효한 객체만 필터링하고 요약된 리포트를 얻고자 한다.

### SRS
- 입력: 추론 결과 리스트 또는 anomaly score map
- 출력: dict 또는 요약 문자열
- 처리:
  - confidence thresholding
  - 검출 수, max score, 평균 score 계산
  - 정렬 및 상위 N개 선택 가능

---

## ✅ 5. visualizer.py

### URS
- 사용자는 결과 이미지를 시각적으로 확인하기 위해 bbox나 heatmap이 overlay된 이미지를 저장하고자 한다.

### SRS
- 입력: `img`, `detections`, `path`
- 처리:
  - bbox/label overlay
  - heatmap blending (alpha=0.5)
  - 저장: `cv2.imwrite`

---

## ✅ 6. main.py

### URS
- 사용자는 명령어 한 줄로 전체 파이프라인을 실행하고 결과를 얻고자 한다.

### SRS
- 처리 흐름:
  - argparse로 폴더 경로 및 옵션 입력
  - 각 모듈 함수 순차 호출
  - 로그 및 결과 저장
- 예외 처리 및 로깅 포함

---

## 🔍 향후 필요 예상 문서

| 모듈명 | 예상 기능 | 필요 여부 |
|--------|-----------|-----------|
| `metrics.py` | F1, precision, recall 계산 | ✅ 필요 |
| `logger.py` | 로깅 유틸리티 | ⛔ (간단한 로깅이면 제외 가능) |
| `image_utils.py` | resize, normalize 유틸 | ⛔ (전처리에 포함될 수 있음) |
| `cli.py` | 명령줄 인자 파서 | ⛔ (main.py에 포함 가능) |
| `web_api.py` | FastAPI 서버 구현 | ✅ (장기 단계에서 필요) |

