"""YOLOv8 모델을 로드하여 이미지 예측을 수행하는 스크립트."""

from pathlib import Path

import cv2
from ultralytics import YOLO

# 테스트용 이미지와 모델 경로를 직접 지정합니다.
TEST_IMAGES_DIR = "./test_images"
MODEL_PATH = "./models/best.pt"  # 학습된 모델 경로 또는 프리트레인 모델


def main() -> None:
    """지정된 폴더의 이미지에 대해 YOLOv8 추론을 실행합니다."""

    print(f"YOLOv8 모델 로드: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, device=0)

    image_paths = sorted(Path(TEST_IMAGES_DIR).glob("*.*"))
    if not image_paths:
        print("이미지를 찾을 수 없습니다.")
        return

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model(img)
        print(f"{img_path} 결과:")
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
