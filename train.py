"""학습 진입점."""
import argparse
import yaml
from modules.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteAOI 학습 스크립트")
    parser.add_argument("--dataset", type=str, help="데이터셋 경로")
    parser.add_argument("--output", type=str, help="모델 저장 경로")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["dataset"] = args.dataset
    config["output_model"] = args.output

    train_model(config)


if __name__ == "__main__":
    main()
