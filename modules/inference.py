from typing import Any

import torch
import numpy as np

_model: Any = None

def load_model(model_path: str, weights_only: bool = False) -> Any:
    """모델을 로드하고 evaluation 모드로 설정합니다.

    Args:
        model_path: 불러올 모델 파일 경로.
        weights_only: ``True`` 면 ``state_dict`` 만 로드합니다.
    """
    global _model
    if weights_only:
        if _model is None:
            raise ValueError(
                "weights_only 옵션을 사용하려면 먼저 모델을 초기화해야 합니다."
            )
        state_dict = torch.load(model_path, map_location="cpu")
        _model.load_state_dict(state_dict)
    else:
        _model = torch.load(model_path, map_location="cpu")
    _model.eval()
    return _model

def run_inference(img: np.ndarray):
    """단순 예시용 추론 함수."""
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    with torch.no_grad():
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return _model(tensor)
