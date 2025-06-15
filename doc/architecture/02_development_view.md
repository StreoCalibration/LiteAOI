# 🔧 Development View

## 디렉토리 구조

시스템 구현은 아래와 같은 폴더 구조를 따릅니다:

```
project_root/
├── yolo_train.py
├── infer.py
├── modules/
│   ├── trainer.py
│   ├── model_loader.py
│   ├── model_downloader.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── inference.py
│   ├── postprocessor.py
│   └── visualizer.py
├── models/
│   └── mymodel_v1.pt
├── datasets/
│   └── ...
├── config.yaml
└── README.md
```

## 기술 스택

- Python 3.10+
- PyTorch, YOLOv5/YOLOv8, Anomalib
- OpenCV, Albumentations
- 설정 파일 기반 실행 (`config.yaml`)

## 구현 세부

- `trainer.py`는 학습 시작 전에 데이터셋 폴더의 `labels.cache` 파일을 자동으로 삭제합니다.
- 모델 로딩과 추론 스크립트는 기본 장치로 GPU를 사용하도록 구성되어 있습니다.
