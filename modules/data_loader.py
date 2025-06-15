"""데이터 로딩을 담당하는 모듈."""
import logging
from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 지원하는 이미지 확장자
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_images(path: str) -> List[str]:
    """지정된 경로에서 이미지 파일 목록을 가져옵니다.
    
    Args:
        path: 이미지 디렉터리 경로
        
    Returns:
        이미지 파일 경로 리스트
        
    Raises:
        FileNotFoundError: 디렉터리가 존재하지 않을 때
        ValueError: 이미지 파일이 없을 때
    """
    try:
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"디렉터리를 찾을 수 없습니다: {path}")
            
        if not path_obj.is_dir():
            raise ValueError(f"경로가 디렉터리가 아닙니다: {path}")
            
        logger.info(f"이미지 로딩 시작: {path}")
        
        # 이미지 파일 검색
        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(path_obj.glob(f"*{ext}"))
            image_files.extend(path_obj.glob(f"*{ext.upper()}"))
            
        # 중복 제거 및 정렬
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {path}")
            
        image_paths = [str(img) for img in image_files]
        logger.info(f"이미지 로딩 완료: {len(image_paths)}개 파일")
        
        return image_paths
        
    except Exception as e:
        logger.error(f"이미지 로딩 실패: {e}")
        raise


def load_single_image(image_path: str) -> np.ndarray:
    """단일 이미지를 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        로드된 이미지 (BGR 형식)
        
    Raises:
        FileNotFoundError: 이미지 파일이 존재하지 않을 때
        ValueError: 이미지 로딩 실패 시
    """
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"이미지 로딩 실패: {image_path}")
            
        return image
        
    except Exception as e:
        logger.error(f"이미지 로딩 실패 ({image_path}): {e}")
        raise


def get_image_info(image_path: str) -> dict:
    """이미지 정보를 반환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        
    Returns:
        이미지 정보 딕셔너리 (크기, 채널 수 등)
    """
    try:
        image = load_single_image(image_path)
        height, width, channels = image.shape
        
        info = {
            "path": image_path,
            "width": width,
            "height": height,
            "channels": channels,
            "size": (width, height),
            "file_size": Path(image_path).stat().st_size
        }
        
        return info
        
    except Exception as e:
        logger.warning(f"이미지 정보 추출 실패 ({image_path}): {e}")
        return {"path": image_path, "error": str(e)}


def validate_image_batch(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """이미지 배치를 검증하고 유효한 파일과 무효한 파일을 분리합니다.
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        
    Returns:
        (유효한 이미지 경로 리스트, 무효한 이미지 경로 리스트)
    """
    valid_images = []
    invalid_images = []
    
    for image_path in image_paths:
        try:
            # 파일 존재 여부 확인
            if not Path(image_path).exists():
                invalid_images.append(image_path)
                continue
                
            # 이미지 로딩 테스트
            image = cv2.imread(image_path)
            if image is None:
                invalid_images.append(image_path)
                continue
                
            valid_images.append(image_path)
            
        except Exception:
            invalid_images.append(image_path)
            
    logger.info(f"이미지 검증 완료: 유효 {len(valid_images)}개, 무효 {len(invalid_images)}개")
    
    if invalid_images:
        logger.warning(f"무효한 이미지 파일들: {invalid_images}")
        
    return valid_images, invalid_images