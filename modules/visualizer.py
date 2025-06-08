from typing import List

import cv2
import numpy as np


def save_result_image(img: np.ndarray, detections: List, path: str) -> None:
    """감지된 영역을 사각형으로 표시하여 저장합니다."""
    output = img.copy()
    for det in detections:
        bbox = det.get("bbox") if isinstance(det, dict) else None
        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(path, output)


def show_image(img: np.ndarray, window_name: str = "Image") -> None:
    """OpenCV 창으로 이미지를 한 번 보여줍니다."""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
