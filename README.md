# LiteAOI

LiteAOI는 단순한 자동 시각 검사 예제 프로젝트입니다. 본 리포지토리는 4+1 뷰로 설계된 아키텍처 문서를 기반으로 기본적인 코드 구조만을 제공합니다.

## 환경 준비

- Python 3.10 이상
 - PyTorch
 - OpenCV
 - Albumentations
 - Anomalib
 - Ultralytics YOLO

필요한 패키지는 `requirements.txt`를 참고하여 설치합니다.

```bash
pip install -r requirements.txt
```

## 사용 방법

### 학습

학습 데이터를 이용하여 모델을 학습하고 싶다면 다음과 같이 실행합니다.

```bash
python train.py --dataset ./datasets/wafer --output ./models/mymodel_v1.pt
```

### YOLOv8 학습

YOLOv8을 이용해 객체 탐지 모델을 학습하려면 `yolo_train.py`를 실행합니다. 데이터셋
구성 파일과 학습 파라미터는 스크립트 안의 상수에서 수정할 수 있습니다.

```bash
python yolo_train.py
```

### 추론

학습된 모델을 사용하여 이미지를 분석하려면 다음 명령을 사용합니다.

```bash
python infer.py --input ./test_images --model ./models/mymodel_v1.pt
```

### 테스트

YOLOv8 모델을 활용해 간단한 테스트를 진행하려면 `test.py`를 실행합니다. 이미지와 모델의 경로는 스크립트 내의 `TEST_IMAGES_DIR`와 `MODEL_PATH` 변수에서 직접 지정합니다.

```bash
python test.py
```

출력된 결과는 콘솔에서 확인할 수 있습니다.

## 폴더 구조

문서의 Development View에서 제시된 구조를 따릅니다.

```text
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
├── datasets/
├── config.yaml
└── README.md
```
