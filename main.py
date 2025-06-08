from pathlib import Path
import yaml
import tkinter as tk

from modules.data_loader import load_images_from_folder
from modules.preprocessor import preprocess_image
from modules.inference import load_model, run_inference
from modules.postprocessor import summarize_detection
from modules.visualizer import save_result_image


class LiteAOIUI:
    """이미지 검사용 간단한 UI 클래스"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.root = tk.Tk()
        self.root.title("LiteAOI")
        tk.Button(
            self.root,
            text="검사 실행",
            command=self.run_detection,
        ).pack(padx=20, pady=20)

    def run_detection(self) -> None:
        data_folder = self.config.get("data_folder", "images")
        model_path = self.config.get("model_path", "models/yolov5s.pt")

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

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    LiteAOIUI().run()
