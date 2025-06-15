"""DeepPCB 데이터셋 준비 스크립트"""
import argparse
import logging
from pathlib import Path
import sys
from modules.deeppcb_loader import DeepPCBLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DeepPCB 데이터셋을 YOLO 형식으로 준비")
    parser.add_argument(
        "--input", 
        default="../DeepPCB",
        help="DeepPCB 데이터셋 경로 (기본값: ../DeepPCB)"
    )
    parser.add_argument(
        "--output",
        default="./datasets/deeppcb",
        help="출력 경로 (기본값: ./datasets/deeppcb)"
    )
    
    args = parser.parse_args()
    
    try:
        # DeepPCB 로더 초기화
        logger.info(f"DeepPCB 데이터셋 로드 중: {args.input}")
        loader = DeepPCBLoader(args.input)
        
        # 데이터셋 준비
        logger.info(f"YOLO 형식으로 변환 중: {args.output}")
        loader.prepare_dataset(args.output)
        
        logger.info("데이터셋 준비 완료!")
        logger.info(f"YAML 파일: {Path(args.output) / 'deeppcb.yaml'}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
