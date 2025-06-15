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

## DeepPCB 데이터셋 사용

### 1. DeepPCB 다운로드

[DeepPCB](https://github.com/tangsanli5201/DeepPCB.git) 리포지터리를 프로젝트와 같은 상위 폴더에 클론합니다.

```bash
git clone https://github.com/tangsanli5201/DeepPCB.git ../DeepPCB
```

### 2. 데이터셋 준비

DeepPCB 데이터셋을 YOLO 형식으로 변환합니다:

```bash
python prepare_deeppcb.py --input ../DeepPCB --output ./datasets/deeppcb
```

이 스크립트는:
- DeepPCB의 `PCBData` 폴더에서 이미지와 라벨을 읽음
- 80%/20% 비율로 train/val 세트로 분할
- YOLO 형식의 디렉터리 구조 생성
- 학습용 YAML 설정 파일 생성 (`datasets/deeppcb/deeppcb.yaml`)

### 3. YOLOv8 학습

준비된 DeepPCB 데이터셋으로 학습:

```bash
python yolo_train.py --data ./datasets/deeppcb/deeppcb.yaml \
                    --model ./models/yolov8x.pt \
                    --epochs 50 --output ./output
```

또는 자동 탐색 기능 사용:

```bash
python yolo_train.py --dataset ../DeepPCB \
                    --model ./models/yolov8x.pt \
                    --epochs 50
```

## DeepPCB 클래스 정보

DeepPCB는 6가지 PCB 결함 유형을 포함합니다:
- 0: open (단선)
- 1: short (단락)
- 2: mousebite (마우스바이트)
- 3: spur (스퍼)
- 4: copper (구리잔여물)
- 5: pin-hole (핀홀)

## 추론

학습된 모델을 사용하여 이미지를 분석:

```bash
python infer.py --input ./test_images --model ./output/best.pt
```

## 테스트

간단한 테스트 실행:

```bash
python test.py
```

## 폴더 구조

```text
project_root/
├── yolo_train.py
├── prepare_deeppcb.py    # DeepPCB 데이터셋 준비 스크립트
├── download_dataset.py
├── infer.py
├── modules/
│   ├── deeppcb_loader.py # DeepPCB 전용 로더
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
│   └── deeppcb/         # 변환된 DeepPCB 데이터셋
├── config.yaml
└── README.md
```
