"""학습된 모델을 로드하는 모듈."""
import logging
from typing import Any, Union
from pathlib import Path
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = "cuda") -> YOLO:
    """모델 파일을 읽어 로드합니다.
    
    Args:
        model_path: 모델 파일 경로 (.pt 파일)
        device: 사용할 장치 ('cuda', 'cpu', 또는 장치 번호)
        
    Returns:
        로드된 YOLO 모델
        
    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않을 때
        RuntimeError: 모델 로딩 실패 시
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        logger.info(f"모델 로딩 시작: {model_path}")
        
        # 장치 설정
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
            device = "cpu"
            
        # YOLO 모델 로드
        model = YOLO(str(model_path))
        model.to(device)
        
        logger.info(f"모델 로딩 완료: {model_path} (device={device})")
        return model
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise RuntimeError(f"모델 로딩 중 오류 발생: {e}")


def load_pretrained(model_path: str, device: str = "cuda") -> YOLO:
    """사전 학습 모델을 로드합니다.
    
    Args:
        model_path: 사전 학습 모델 경로
        device: 사용할 장치
        
    Returns:
        로드된 사전 학습 YOLO 모델
    """
    logger.info(f"사전 학습 모델 로딩: {model_path}")
    return load_model(model_path, device)


def get_model_info(model: YOLO) -> dict:
    """모델 정보를 반환합니다.
    
    Args:
        model: YOLO 모델
        
    Returns:
        모델 정보 딕셔너리
    """
    try:
        info = {
            "model_type": type(model).__name__,
            "device": str(model.device) if hasattr(model, 'device') else "unknown",
            "num_classes": getattr(model.model, 'nc', 'unknown') if hasattr(model, 'model') else 'unknown',
            "input_size": getattr(model.model, 'imgsz', 'unknown') if hasattr(model, 'model') else 'unknown'
        }
        return info
    except Exception as e:
        logger.warning(f"모델 정보 추출 실패: {e}")
        return {"error": str(e)}