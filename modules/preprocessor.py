import cv2
import numpy as np

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """이미지를 흑백으로 변환 후 히스토그램 평활화를 수행합니다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = cv2.equalizeHist(gray)
    return normalized
