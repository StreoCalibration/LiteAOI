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

데이터셋은 기본적으로 제공되지 않으므로 사용 전에 적절한 데이터 폴더를 준비해야
합니다. 예시로 [DeepPCB](https://github.com/tangsanli5201/DeepPCB.git)
리포지터리를 프로젝트와 같은 상위 폴더에 클론해 사용할 수 있습니다.

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git ../DeepPCB
```
클론한 리포지토리의 `PCBData` 폴더 안에는 여러 세트가 포함되어 있으며 각 세트마다 이미지와 라벨 구성 파일(`*.txt`)이 있습니다. 학습에 사용할 세트 경로를 `--dataset` 인자로 지정하면 됩니다. `yolo_train.py`에서 `--data` 옵션을 생략하면 이 위치에서 YAML 파일을 자동으로 찾으며 `test.py` 또한 동일한 규칙으로 동작합니다.



### YOLOv8 학습

YOLOv8을 이용해 객체 탐지 모델을 학습하려면 `yolo_train.py`를 실행합니다. 데이터셋
구성 파일, 사전 학습 모델, 에폭 수와 출력 디렉터리를 인자로 전달해 설정할 수 있습니다.
`labels.cache` 파일이 존재할 경우 스크립트가 자동으로 삭제하며, 기본 장치는 GPU로 설정되어 있습니다.

```bash
python yolo_train.py --data ./datasets/dataset.yaml \
                    --model ./models/yolov8x.pt \
                    --epochs 50 --output ./output
```

`--data` 옵션을 생략하면 프로젝트 상위 폴더의 `DeepPCB` 리포지터리에서
YOLO 형식의 YAML 파일을 자동으로 탐색합니다. 특정 경로를 사용하려면
`--dataset` 옵션에 DeepPCB 하위 폴더를 지정할 수 있습니다.

`--data`와 `--dataset`을 함께 지정할 경우 `--data`에 입력한 YAML 경로가
존재하면 그 파일을 우선 사용합니다. 경로가 잘못되었거나 옵션을 생략하면
`--dataset`에서 지정한 폴더를 검색해 YAML 파일을 찾습니다. 두 옵션 모두
없다면 `datasets/dataset.yaml`이 기본값으로 사용됩니다.

예를 들어, 다음 명령은 `coco128.yaml` 파일이 현재 폴더에 있을 경우 해당
데이터셋으로 학습하며, 파일이 없으면 `../DeepPCB` 폴더에서 YAML을
찾아 사용합니다.

```bash
python yolo_train.py \
    --data coco128.yaml \
    --dataset ../DeepPCB \
    --model models/yolov8x.pt \
    --epochs 5 \
    --output output
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
├── yolo_train.py
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
