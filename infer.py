"""추론 진입점."""
import argparse
import logging
import sys
from pathlib import Path
import yaml
from modules.model_loader import load_model
from modules.data_loader import load_images
from modules.preprocessor import preprocess
from modules.inference import run_inference
from modules.postprocessor import summarize
from modules.visualizer import visualize

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """설정 파일을 로드합니다."""
    try:
        if not Path(config_path).exists():
            logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
            return {}
            
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"설정 파일 파싱 오류: {e}")
        return {}
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류: {e}")
        return {}


def validate_paths(input_dir: str, model_path: str) -> bool:
    """입력 경로들을 검증합니다."""
    if not Path(input_dir).exists():
        logger.error(f"입력 디렉터리를 찾을 수 없습니다: {input_dir}")
        return False
        
    if not Path(model_path).exists():
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return False
        
    # 입력 디렉터리에 이미지 파일이 있는지 확인
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in Path(input_dir).iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"입력 디렉터리에 이미지 파일이 없습니다: {input_dir}")
        return False
        
    logger.info(f"발견된 이미지 파일 수: {len(image_files)}")
    return True


def main() -> None:
    """추론을 실행합니다."""
    try:
        parser = argparse.ArgumentParser(description="LiteAOI 추론 스크립트")
        parser.add_argument("--input", type=str, help="입력 이미지 디렉터리")
        parser.add_argument("--model", type=str, help="모델 파일 경로")
        parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
        parser.add_argument("--save", default=None, help="결과 저장 폴더")
        parser.add_argument("--confidence", type=float, default=None, help="신뢰도 임계값")
        args = parser.parse_args()

        # 설정 로드
        config = load_config(args.config)
        inference_config = config.get("inference", {})
        model_config = config.get("model", {})

        # 기본값 설정
        input_dir = args.input or inference_config.get("input_dir", "./test_images")
        model_path = args.model or model_config.get("output", "./models/mymodel_v1.pt")
        save_dir = args.save or inference_config.get("output_dir", "./results")
        confidence = args.confidence or inference_config.get("confidence", 0.5)
        device = model_config.get("device", "cuda")

        logger.info("=== LiteAOI 추론 시작 ===")
        logger.info(f"입력 디렉터리: {input_dir}")
        logger.info(f"모델 경로: {model_path}")
        logger.info(f"결과 저장: {save_dir}")
        logger.info(f"신뢰도 임계값: {confidence}")
        logger.info(f"장치: {device}")

        # 경로 검증
        if not validate_paths(input_dir, model_path):
            sys.exit(1)

        # 결과 저장 디렉터리 생성
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 추론 파이프라인 실행
        logger.info("모델 로딩...")
        model = load_model(model_path, device=device)
        
        logger.info("이미지 로딩...")
        images = load_images(input_dir)
        
        logger.info("이미지 전처리...")
        processed = preprocess(images)
        
        logger.info("추론 실행...")
        results = run_inference(model, processed)
        
        logger.info("결과 후처리...")
        summarized = summarize(results)
        
        if inference_config.get("visualize", True):
            logger.info("결과 시각화...")
            visualize(summarized, save_dir)
        
        logger.info("=== 추론 완료 ===")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 추론이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"추론 중 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()