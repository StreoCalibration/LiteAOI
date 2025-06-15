"""YOLOv8 모델을 사용자 데이터셋으로 학습하는 스크립트."""

from pathlib import Path
import shutil
from ultralytics import YOLO

# 데이터셋 설정(YOLO 포맷 YAML)
DATA_CONFIG = str(Path(__file__).resolve().parent / "datasets" / "dataset.yaml")
# 높은 성능을 위해 yolov8x 모델을 사용합니다.
PRETRAINED_MODEL = "./models/yolov8x.pt"
EPOCHS = 50
# 학습된 모델이 저장될 디렉터리
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    """YOLOv8 훈련을 실행합니다."""

    print(f"프리트레인 모델 로드: {PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)

    print(f"데이터셋: {DATA_CONFIG}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        project=str(OUTPUT_DIR),
        name="yolo_custom",
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"학습 완료. 최종 모델: {best_path}")

    final_path = OUTPUT_DIR / "best.pt"
    shutil.copy(best_path, final_path)
    print(f"베스트 모델을 {final_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
