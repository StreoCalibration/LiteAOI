# 🔧 Development View

## 디렉토리 구조

시스템 구현은 아래와 같은 폴더 구조를 따릅니다:

```
project_root/
├── train.py
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
