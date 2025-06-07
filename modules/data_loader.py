from pathlib import Path
from typing import List

import cv2
import numpy as np

def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """폴더 내의 이미지 파일을 모두 읽어 리스트로 반환합니다."""
    images = []
    folder = Path(folder_path)
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for img_path in folder.glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
    return images
