"""YOLOv8 모델을 로드하여 이미지 예측을 수행하는 스크립트."""

from pathlib import Path

import cv2
from ultralytics import YOLO

# 기본 테스트 이미지 폴더.
# 프로젝트와 같은 상위 경로에 존재하는 DeepPCB 리포지토리를 우선 사용합니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DEEPPCB_ROOT = (PROJECT_ROOT / ".." / "DeepPCB").resolve()

_candidate_dirs = [
    DEEPPCB_ROOT / "dataset" / "test" / "images",
    DEEPPCB_ROOT / "datasets" / "test" / "images",
    DEEPPCB_ROOT / "PCBData" / "test" / "images",
]

TEST_IMAGES_DIR = None
for _d in _candidate_dirs:
    if _d.exists():
        TEST_IMAGES_DIR = str(_d)
        break

if TEST_IMAGES_DIR is None:
    # 폴더가 존재하지 않으면 기본 경로를 사용합니다.
    TEST_IMAGES_DIR = str(PROJECT_ROOT / "test_images")

MODEL_PATH = "./models/best.pt"  # 학습된 모델 경로 또는 프리트레인 모델


def main() -> None:
    """지정된 폴더의 이미지에 대해 YOLOv8 추론을 실행합니다."""

    print(f"YOLOv8 모델 로드: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        print(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
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
