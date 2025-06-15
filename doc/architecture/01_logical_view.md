# 🧩 Logical View

## 구조 개요

이 뷰는 시스템이 어떤 기능을 수행하는지 기능 단위로 나누어 설명합니다. 각 모듈은 하나의 책임만을 가지며, 학습과 추론 파이프라인은 완전히 분리됩니다.

```
train.py                # 학습 진입점
infer.py                # 추론 진입점
├── model_loader.py     # 학습된 모델 로딩
├── data_loader.py      # 이미지 로딩
├── preprocessor.py     # 이미지 전처리
├── inference.py        # 추론 수행
├── postprocessor.py    # 결과 요약 및 필터링
└── visualizer.py       # 시각화 및 저장
```

## 주요 기능 흐름

- `train.py`: 학습용 데이터셋을 불러오기 전에 `labels.cache` 파일을 삭제하고 모델을 학습합니다.
- `infer.py`: 저장된 모델을 불러와 입력 이미지에 대해 GPU를 기본으로 사용해 추론을 수행합니다.
