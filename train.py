"""학습 진입점."""
import argparse
import yaml
from pathlib import Path
from modules.trainer import train_model
from modules.dataset_downloader import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteAOI 학습 스크립트")
    parser.add_argument("--dataset", type=str, help="데이터셋 경로")
    parser.add_argument("--output", type=str, help="모델 저장 경로")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--pretrained", default=None, help="사전 학습 모델 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dataset"] = args.dataset
    config["output_model"] = args.output
    if args.pretrained:
        config["pretrained_model"] = args.pretrained

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        url = config.get("dataset_url")
        if url:
            download_dataset(url, str(dataset_path))

    train_model(config)


if __name__ == "__main__":
    main()

