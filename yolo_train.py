"""YOLOv8 모델을 사용자 데이터셋으로 학습하는 스크립트."""

import logging
import sys
from pathlib import Path
import shutil
import argparse
import yaml
from typing import Optional
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"설정 파일 파싱 오류: {e}")
        return {}


def find_yaml_file(directory: Path) -> Optional[Path]:
    """주어진 디렉터리에서 첫 번째 YAML 파일을 반환합니다."""
    try:
        for ext in ("*.yaml", "*.yml"):
            candidates = list(directory.glob(ext))
            if candidates:
                return candidates[0]
        return None
    except Exception as e:
        logger.error(f"YAML 파일 검색 중 오류: {e}")
        return None


def find_deeppcb_data(dataset_dir: Optional[str]) -> Optional[str]:
    """DeepPCB 경로에서 학습용 YAML 파일을 탐색합니다."""
    try:
        if dataset_dir:
            d = Path(dataset_dir)
            if d.is_file():
                return str(d)
            if d.is_dir():
                yaml_path = find_yaml_file(d)
                if yaml_path:
                    return str(yaml_path)
                for sub in sorted(d.iterdir()):
                    if sub.is_dir():
                        yaml_path = find_yaml_file(sub)
                        if yaml_path:
                            return str(yaml_path)
            return None

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
            yaml_path = find_yaml_file(root)
            if yaml_path:
                logger.info(f"DeepPCB 데이터셋 발견: {yaml_path}")
                return str(yaml_path)
            if root.name == "PCBData":
                subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
                for sub in subdirs:
                    yaml_path = find_yaml_file(sub)
                    if yaml_path:
                        logger.info(f"DeepPCB 데이터셋 발견: {yaml_path}")
                        return str(yaml_path)
        return None
    except Exception as e:
        logger.error(f"DeepPCB 데이터셋 검색 중 오류: {e}")
        return None


def remove_labels_cache(data_config: str) -> None:
    """데이터셋 캐시 파일이 존재하면 삭제합니다."""
    try:
        cache_path = Path(data_config).resolve().parent / "labels.cache"
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"캐시 파일 삭제: {cache_path}")
    except Exception as e:
        logger.error(f"캐시 파일 삭제 중 오류: {e}")


def validate_model_path(model_path: str) -> bool:
    """모델 파일 경로를 검증합니다."""
    if not Path(model_path).exists():
        logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    return True


def main() -> None:
    """YOLOv8 훈련을 실행합니다."""
    try:
        # 설정 로드
        config = load_config()
        
        # 기본값 설정 - config.yaml에서 먼저 읽기
        DATA_CONFIG_DEFAULT = config.get("dataset", {}).get("path", "./datasets/deeppcb/deeppcb.yaml")
        PRETRAINED_MODEL_DEFAULT = config.get("model", {}).get("pretrained", "./models/yolov8x.pt")
        EPOCHS_DEFAULT = config.get("training", {}).get("epochs", 50)
        OUTPUT_DIR_DEFAULT = Path(config.get("training", {}).get("project", "./output"))

        parser = argparse.ArgumentParser(description="YOLOv8 학습 스크립트")
        parser.add_argument("--data", default=None, help="데이터셋 YAML 경로")
        parser.add_argument("--dataset", default=None, help="DeepPCB 데이터셋 경로")
        parser.add_argument("--model", default=PRETRAINED_MODEL_DEFAULT, help="사전 학습 모델 경로")
        parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="학습 에폭 수")
        parser.add_argument("--output", default=str(OUTPUT_DIR_DEFAULT), help="결과 저장 디렉터리")
        parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
        args = parser.parse_args()

        # 데이터셋 경로 결정
        data_config = args.data
        if not data_config:
            # 1. 준비된 DeepPCB 데이터셋 확인
            prepared_deeppcb = Path("./datasets/deeppcb/deeppcb.yaml")
            if prepared_deeppcb.exists():
                data_config = str(prepared_deeppcb)
                logger.info("준비된 DeepPCB 데이터셋 사용")
            else:
                # 2. 원본 DeepPCB에서 찾기
                data_config = find_deeppcb_data(args.dataset)
                if not data_config:
                    # 3. config.yaml의 기본값 사용
                    data_config = DATA_CONFIG_DEFAULT
                    
        # 데이터셋 검증
        if not Path(data_config).exists():
            logger.error(f"데이터셋 파일을 찾을 수 없습니다: {data_config}")
            logger.info("💡 먼저 DeepPCB 데이터셋을 준비하세요:")
            logger.info("   python prepare_deeppcb.py")
            sys.exit(1)
            
        logger.info(f"데이터셋: {data_config}")
        
        # 캐시 삭제
        remove_labels_cache(data_config)

        # 모델 검증 및 로드
        if not validate_model_path(args.model):
            sys.exit(1)
            
        logger.info(f"사전 학습 모델 로드: {args.model}")
        model = YOLO(args.model)
        
        # 출력 디렉터리 생성
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습 실행
        logger.info("모델 학습 시작...")
        results = model.train(
            data=data_config,
            epochs=args.epochs,
            project=str(output_dir),
            name=config.get("training", {}).get("name", "deeppcb_yolo"),
            device=config.get("model", {}).get("device", 0),
        )

        # 결과 저장
        best_path = Path(results.save_dir) / "weights" / "best.pt"
        final_path = output_dir / "best.pt"
        
        if best_path.exists():
            shutil.copy(best_path, final_path)
            logger.info(f"학습 완료! 최종 모델: {final_path}")
        else:
            logger.error("학습된 모델 파일을 찾을 수 없습니다.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 학습이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"학습 중 예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
