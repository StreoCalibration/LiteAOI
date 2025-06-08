from typing import Any

import torch
import numpy as np

_model: Any = None

def load_model(model_path):
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    return model

def run_inference(img: np.ndarray, _model):
    """단순 예시용 추론 함수."""
    if _model is None:
        raise RuntimeError("모델이 로드되지 않았습니다.")
    with torch.no_grad():
        tensor = torch.from_numpy(img).unsqueeze(0).float()
        return _model(tensor)
