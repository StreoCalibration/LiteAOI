"""모델 학습을 담당하는 모듈."""
from typing import Any, Dict
import os
from modules.model_loader import load_pretrained


def train_model(config: Dict[str, Any]) -> None:
    """주어진 설정으로 모델을 학습하고 저장합니다.

    Args:
        config: 학습에 필요한 설정 딕셔너리. ``output_model`` 경로가 포함되어야 합니다.
    """
    dataset_path = config.get("dataset")
    output_model = config.get("output_model")
    pretrained_model = config.get("pretrained_model")

    if pretrained_model and os.path.exists(pretrained_model):
        load_pretrained(pretrained_model)
    elif pretrained_model:
        print("사전 학습 모델이 존재하지 않습니다: " + pretrained_model)

    # 실제 학습 로직은 생략되어 있습니다.
    print(f"데이터셋 로드: {dataset_path}")
    print("모델 학습 중 ...")
    print(f"학습된 모델 저장: {output_model}")
