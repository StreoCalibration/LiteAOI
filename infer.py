"""추론 진입점."""
import argparse
import yaml
from modules.model_loader import load_model
from modules.data_loader import load_images
from modules.preprocessor import preprocess
from modules.inference import run_inference
from modules.postprocessor import summarize
from modules.visualizer import visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="LiteAOI 추론 스크립트")
    parser.add_argument("--input", type=str, help="입력 이미지 디렉터리")
    parser.add_argument("--model", type=str, help="모델 파일 경로")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--save", default="./results", help="결과 저장 폴더")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = load_model(args.model, device="cuda")
    images = load_images(args.input)
    processed = preprocess(images)
    results = run_inference(model, processed)
    summarized = summarize(results)
    visualize(summarized, args.save)


if __name__ == "__main__":
    main()
