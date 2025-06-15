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

### 데이터셋 준비

데이터가 포함되어 있지 않으므로 학습이나 테스트 전에 아래 스크립트를 사용해
필요한 데이터셋을 내려받을 수 있습니다.

```bash
python download_dataset.py https://github.com/example/wafer-dataset.git \
    --output ./datasets/wafer
```

추가로, [DeepPCB](https://github.com/tangsanli5201/DeepPCB.git) 리포지토리를
프로젝트와 같은 상위 폴더에 클론해 사용할 수 있습니다.

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git ../DeepPCB
```

`test.py` 스크립트는 위 경로가 존재하면 자동으로 DeepPCB의 테스트 이미지를
사용합니다. 경로가 없을 경우에는 기본 `test_images` 폴더를 참조합니다.
`train.py`에서도 `--dataset` 옵션을 주지 않으면 해당 위치의 학습 데이터를
자동으로 사용합니다.

## 사용 방법

### 학습

학습 스크립트는 개발자용으로 제공되며, 프로젝트 폴더에 위치한 사전 학습 모델을
불러올 수 있습니다. 다음과 같이 실행합니다.

```bash
python train.py --output ./models/mymodel_v1.pt \
    --pretrained ./models/pretrained.pt
```
`--dataset` 인자를 생략하면 `../DeepPCB` 폴더가 존재할 경우 그 안의
데이터셋을 자동으로 사용합니다. 경로가 없으면 기본 설정값을 따릅니다.

### YOLOv8 학습

YOLOv8을 이용해 객체 탐지 모델을 학습하려면 `yolo_train.py`를 실행합니다. 데이터셋
구성 파일, 사전 학습 모델, 에폭 수와 출력 디렉터리를 인자로 전달해 설정할 수 있습니다.
`labels.cache` 파일이 존재할 경우 스크립트가 자동으로 삭제하며, 기본 장치는 GPU로 설정되어 있습니다.

```bash
python yolo_train.py --data ./datasets/dataset.yaml \
                    --model ./models/yolov8x.pt \
                    --epochs 50 --output ./output
```

### 추론

학습된 모델을 사용하여 이미지를 분석하려면 다음 명령을 사용합니다.

기본적으로 GPU를 사용해 추론을 수행합니다.

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
├── download_dataset.py
├── infer.py
├── modules/
│   ├── trainer.py
│   ├── model_loader.py
│   ├── model_downloader.py
│   ├── dataset_downloader.py
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
