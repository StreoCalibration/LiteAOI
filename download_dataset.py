"""데이터셋을 다운로드하는 스크립트."""
import argparse
from modules.dataset_downloader import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터셋 다운로드")
    parser.add_argument("url", help="데이터셋 URL")
    parser.add_argument(
        "--output",
        default="./datasets/wafer",
        help="데이터셋을 저장할 경로",
    )
    args = parser.parse_args()

    download_dataset(args.url, args.output)


if __name__ == "__main__":
    main()

