# ✅ Scenario View

## 유스케이스 1: 학습 실행

```
python train.py --dataset ./datasets/wafer --output ./models/mymodel_v1.pt
```
학습 시작 시 데이터셋 폴더의 `labels.cache` 파일이 자동으로 삭제되며,
모델 학습은 기본적으로 GPU에서 수행됩니다.

## 유스케이스 2: 추론 실행

```
python infer.py --input ./test_images --model ./models/mymodel_v1.pt
```
추론 또한 GPU를 기본 장치로 사용합니다.

## Codex 연동 가이드

Codex는 각 파일별로 역할이 명확하게 분리되어 있기 때문에, 각 모듈의 주석 또는 함수 시그니처를 바탕으로 자동 완성 및 구현 보조가 가능합니다.

- 함수 기반 설계: Codex는 `def`를 기반으로 내부 구현을 쉽게 유추함
- 예시 기반 학습: `train.py`, `infer.py`의 전체 흐름을 파악하여 Codex는 순차적 구현 제안 가능
- `config.yaml`을 읽는 패턴은 반복되므로 자동 완성이 용이함

**팁:** 각 파일마다 Docstring을 명확히 작성하면 Codex가 의미를 더 잘 파악하고 보완 코드를 제안합니다.
