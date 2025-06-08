from pathlib import Path
import yaml

from modules.data_loader import load_images_from_folder
from modules.preprocessor import preprocess_image
from modules.inference import load_model, run_inference
from modules.postprocessor import summarize_detection
from modules.visualizer import save_result_image


def main(config_path: str = "config.yaml") -> None:
    config = yaml.safe_load(Path(config_path).read_text())
    data_folder = config.get("data_folder", "images")
    model_path = config.get("model_path", "models/yolov5s.pt")

    images = load_images_from_folder(data_folder)
    if not images:
        print("이미지를 찾을 수 없습니다.")
        return

    model = load_model(model_path)

    for idx, img in enumerate(images):
        processed = preprocess_image(img)
        results = run_inference(processed)
        summary = summarize_detection(results)
        print(f"[Image {idx}] {summary}")
        save_result_image(img, results, f"output_{idx}.jpg")


if __name__ == "__main__":
    main()
