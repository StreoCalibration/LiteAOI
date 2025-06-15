"""YOLOv8 모델을 로드하여 이미지 예측을 수행하는 스크립트."""

from pathlib import Path
import cv2
from ultralytics import YOLO
import yaml

# DeepPCB 클래스 이름
DEEPPCB_CLASSES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole"
}

# 기본 테스트 이미지 폴더.
PROJECT_ROOT = Path(__file__).resolve().parent
DEEPPCB_ROOT = (PROJECT_ROOT / ".." / "DeepPCB").resolve()

_candidate_dirs = [
    DEEPPCB_ROOT / "PCBData" / "group00000" / "group00000",  # DeepPCB 첫 번째 그룹
    DEEPPCB_ROOT / "dataset" / "test" / "images",
    DEEPPCB_ROOT / "datasets" / "test" / "images",
    PROJECT_ROOT / "datasets" / "deeppcb" / "images" / "val",  # 준비된 데이터셋의 검증 세트
]

TEST_IMAGES_DIR = None
for _d in _candidate_dirs:
    if _d.exists():
        TEST_IMAGES_DIR = str(_d)
        print(f"테스트 이미지 경로: {TEST_IMAGES_DIR}")
        break

if TEST_IMAGES_DIR is None:
    TEST_IMAGES_DIR = str(PROJECT_ROOT / "test_images")
    print(f"기본 테스트 이미지 경로 사용: {TEST_IMAGES_DIR}")

# 설정 파일에서 모델 경로 가져오기
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        MODEL_PATH = config.get("model", {}).get("output", "./models/deeppcb_best.pt")
except:
    MODEL_PATH = "./models/deeppcb_best.pt"

# 모델이 없으면 학습된 결과에서 찾기
if not Path(MODEL_PATH).exists():
    output_best = PROJECT_ROOT / "output" / "deeppcb_yolo" / "weights" / "best.pt"
    if output_best.exists():
        MODEL_PATH = str(output_best)
    else:
        MODEL_PATH = "./models/yolov8x.pt"  # 사전 학습 모델


def main() -> None:
    """지정된 폴더의 이미지에 대해 YOLOv8 추론을 실행합니다."""
    
    print(f"\n=== DeepPCB 테스트 시작 ===")
    print(f"모델 경로: {MODEL_PATH}")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("💡 먼저 학습을 실행하세요: python yolo_train.py")
        return
        
    model = YOLO(MODEL_PATH)
    print(f"✅ 모델 로드 완료")
    
    image_paths = sorted(Path(TEST_IMAGES_DIR).glob("*.jpg"))
    if not image_paths:
        image_paths = sorted(Path(TEST_IMAGES_DIR).glob("*.JPG"))
    
    if not image_paths:
        print(f"❌ 이미지를 찾을 수 없습니다: {TEST_IMAGES_DIR}")
        return
        
    print(f"📁 {len(image_paths)}개 이미지 발견\n")
    
    # 각 이미지에 대해 추론 수행
    for idx, img_path in enumerate(image_paths[:5]):  # 처음 5개만 테스트
        print(f"\n[{idx+1}/{min(5, len(image_paths))}] {img_path.name}")
        print("-" * 50)
        
        img = cv2.imread(str(img_path))
        if img is None:
            print("❌ 이미지 로드 실패")
            continue
            
        results = model(img, verbose=False)
        
        # 결과 표시
        for r in results:
            if len(r.boxes) == 0:
                print("✅ 결함 없음")
            else:
                print(f"⚠️  {len(r.boxes)}개 결함 감지:")
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = DEEPPCB_CLASSES.get(cls, f"Unknown({cls})")
                    print(f"   - {class_name}: {conf:.2%} 신뢰도")
                    
        # 결과 시각화 (선택적)
        # annotated = results[0].plot()
        # cv2.imshow("Result", annotated)
        # cv2.waitKey(0)
    
    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    main()
