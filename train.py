"""학습 진입점."""
import argparse
import yaml
from pathlib import Path
from modules.trainer import train_model
from modules.dataset_downloader import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteAOI 학습 스크립트")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="데이터셋 경로 (기본: DeepPCB가 존재하면 해당 경로 사용)",
    )
    parser.add_argument("--output", type=str, help="모델 저장 경로")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--pretrained", default=None, help="사전 학습 모델 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 데이터셋 경로 설정
    dataset_path = Path(args.dataset) if args.dataset else None

    if dataset_path is None:
        # 프로젝트 상위 폴더에 클론된 DeepPCB가 있는지 확인합니다.
        project_root = Path(__file__).resolve().parent
        deeppcb_root = (project_root / ".." / "DeepPCB").resolve()
        candidate_roots = [
            deeppcb_root / "dataset",
            deeppcb_root / "datasets",
            deeppcb_root / "PCBData",
        ]
        for root in candidate_roots:
            if not root.exists():
                continue
            # PCBData 폴더 안에 여러 세트가 있을 경우 첫 번째 세트를 선택합니다.
            if root.name == "PCBData":
                subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
                if subdirs:
                    dataset_path = subdirs[0]
                    print(f"DeepPCB 데이터셋 사용: {dataset_path}")
                    break
                txt_files = sorted(root.glob("*.txt"))
                if txt_files:
                    candidate = root / txt_files[0].stem
                    if candidate.exists():
                        dataset_path = candidate
                        print(f"DeepPCB 데이터셋 사용: {dataset_path}")
                        break
            if (root / "train").exists() or (root / "train" / "images").exists():
                dataset_path = root
                print(f"DeepPCB 데이터셋 사용: {dataset_path}")
                break

    if dataset_path is None:
        dataset_path = Path(config.get("dataset", "./datasets/wafer"))

    config["dataset"] = str(dataset_path)
    config["output_model"] = args.output
    if args.pretrained:
        config["pretrained_model"] = args.pretrained

    if not dataset_path.exists():
        url = config.get("dataset_url")
        if url:
            download_dataset(url, str(dataset_path))

    train_model(config)


if __name__ == "__main__":
    main()

