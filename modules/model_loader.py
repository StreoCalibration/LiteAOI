"""학습된 모델을 로드하는 모듈."""
from typing import Any


def load_model(model_path: str, device: str = "cuda") -> Any:
    """모델 파일을 읽어 로드합니다."""
    print(f"모델 로드: {model_path} (device={device})")
    # 실제 로딩 로직은 생략
    return None


def load_pretrained(model_path: str, device: str = "cuda") -> Any:
    """사전 학습 모델을 로드합니다."""
    print(f"사전 학습 모델 로드: {model_path} (device={device})")
    return load_model(model_path, device)
