"""이미지 로더 모듈."""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """폴더 내의 이미지 파일을 모두 읽어 리스트로 반환합니다.

    지정된 폴더가 존재하지 않으면 ``FileNotFoundError`` 를 발생시킵니다. 지원되지
    않는 확장자는 무시하며, 이미지 로딩에 실패한 경우 경고 메시지만 출력합니다.
    반환되는 리스트의 순서는 파일명 기준 오름차순으로 유지됩니다.
    """

    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")

    images: List[np.ndarray] = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(SUPPORTED_EXTS):
            continue
        img_path = folder / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"경고: 이미지를 불러오지 못했습니다: {img_path}")
            continue
        images.append(img)

    if not images:
        print("경고: 유효한 이미지가 없습니다.")

    return images
