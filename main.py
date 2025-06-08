from pathlib import Path
import yaml
import tkinter as tk
from tkinter import filedialog

from modules.data_loader import load_images_from_folder
from modules.preprocessor import preprocess_image
from modules.inference import load_model, run_inference
from modules.postprocessor import summarize_detection
from modules.visualizer import save_result_image


class LiteAOIUI:
    """이미지 검사용 간단한 UI 클래스"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config_path = Path(config_path)
        if self.config_path.exists():
            self.config = yaml.safe_load(self.config_path.read_text()) or {}
        else:
            self.config = {}

        self.root = tk.Tk()
        self.root.title("LiteAOI")

        self.data_folder_var = tk.StringVar(
            value=self.config.get("data_folder", "images")
        )
        self.model_path_var = tk.StringVar(
            value=self.config.get("model_path", "models/yolov5s.pt")
        )

        tk.Label(self.root, text="이미지 폴더:").pack(anchor="w", padx=10, pady=2)
        tk.Entry(self.root, textvariable=self.data_folder_var, width=40).pack(
            fill="x", padx=10
        )
        tk.Button(
            self.root, text="폴더 선택", command=self.select_data_folder
        ).pack(padx=10, pady=5)

        tk.Label(self.root, text="모델 파일:").pack(anchor="w", padx=10, pady=2)
        tk.Entry(self.root, textvariable=self.model_path_var, width=40).pack(
            fill="x", padx=10
        )
        tk.Button(
            self.root, text="모델 선택", command=self.select_model_file
        ).pack(padx=10, pady=5)

        tk.Button(
            self.root, text="검사 실행", command=self.run_detection
        ).pack(padx=20, pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_data_folder(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.data_folder_var.get())
        if folder:
            self.data_folder_var.set(folder)
            self.save_config()

    def select_model_file(self) -> None:
        initial_dir = str(Path(self.model_path_var.get()).parent)
        file = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("PyTorch Model", "*.pt *.pth"), ("All files", "*.*")],
        )
        if file:
            self.model_path_var.set(file)
            self.save_config()

    def save_config(self) -> None:
        self.config["data_folder"] = self.data_folder_var.get()
        self.config["model_path"] = self.model_path_var.get()
        self.config_path.write_text(
            yaml.safe_dump(self.config, allow_unicode=True)
        )

    def on_close(self) -> None:
        self.save_config()
        self.root.destroy()

    def run_detection(self) -> None:
        data_folder = self.data_folder_var.get()
        model_path = self.model_path_var.get()

        images = load_images_from_folder(data_folder)
        if not images:
            print("이미지를 찾을 수 없습니다.")
            return

        model = load_model(model_path)

        for idx, img in enumerate(images):
            processed = preprocess_image(img)
            results = run_inference(processed, model)
            summary = summarize_detection(results)
            print(f"[Image {idx}] {summary}")
            save_result_image(img, results, f"output_{idx}.jpg")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    LiteAOIUI().run()
