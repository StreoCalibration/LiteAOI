"""YOLOv8 모델을 사용자 데이터셋으로 학습하는 스크립트."""

from pathlib import Path
import shutil
import argparse
from ultralytics import YOLO


def remove_labels_cache(data_config: str) -> None:
    """데이터셋 캐시 파일이 존재하면 삭제합니다."""
    cache_path = Path(data_config).resolve().parent / "labels.cache"
    if cache_path.exists():
        cache_path.unlink()
        print(f"캐시 파일 삭제: {cache_path}")

# 데이터셋 설정(YOLO 포맷 YAML)
DATA_CONFIG_DEFAULT = str(Path(__file__).resolve().parent / "datasets" / "dataset.yaml")
# 높은 성능을 위해 yolov8x 모델을 사용합니다.
PRETRAINED_MODEL_DEFAULT = "./models/yolov8x.pt"
EPOCHS_DEFAULT = 50
# 학습된 모델이 저장될 디렉터리
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "output"


def main() -> None:
    """YOLOv8 훈련을 실행합니다."""

    parser = argparse.ArgumentParser(description="YOLOv8 학습 스크립트")
    parser.add_argument(
        "--data",
        default=DATA_CONFIG_DEFAULT,
        help="데이터셋 YAML 경로",
    )
    parser.add_argument(
        "--model",
        default=PRETRAINED_MODEL_DEFAULT,
        help="사전 학습 모델 경로",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT,
        help="학습 에폭 수",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR_DEFAULT),
        help="결과 저장 디렉터리",
    )
    args = parser.parse_args()

    print(f"프리트레인 모델 로드: {args.model}")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"프리트레인 모델을 찾을 수 없습니다: {args.model}")
    model = YOLO(args.model)

    print(f"데이터셋: {args.data}")
    remove_labels_cache(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        project=str(output_dir),
        name="yolo_custom",
        device=0,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"학습 완료. 최종 모델: {best_path}")

    final_path = output_dir / "best.pt"
    shutil.copy(best_path, final_path)
    print(f"베스트 모델을 {final_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
