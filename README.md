# LiteAOI

LiteAOI는 단순한 자동 시각 검사 예제 프로젝트입니다. 본 리포지토리는 4+1 뷰로 설계된 아키텍처 문서를 기반으로 기본적인 코드 구조만을 제공합니다.

## 환경 준비

- Python 3.10 이상
- PyTorch
- OpenCV
- Albumentations
- Anomalib

필요한 패키지는 `requirements.txt`를 참고하여 설치합니다.

```bash
pip install -r requirements.txt
```

## 사용 방법

### 학습

학습 스크립트는 개발자용으로 제공되며, 프로젝트 폴더에 위치한 사전 학습 모델을
불러올 수 있습니다. 다음과 같이 실행합니다.

```bash
python train.py --dataset ./datasets/wafer --output ./models/mymodel_v1.pt \
    --pretrained ./models/pretrained.pt
```

### 추론

학습된 모델을 사용하여 이미지를 분석하려면 다음 명령을 사용합니다.

```bash
python infer.py --input ./test_images --model ./models/mymodel_v1.pt
```

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
