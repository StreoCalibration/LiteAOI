"""DeepPCB 데이터셋 로더 모듈"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class DeepPCBLoader:
    """DeepPCB 데이터셋을 로드하고 처리하는 클래스"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.validate_dataset()
        
    def validate_dataset(self):
        """데이터셋 구조 검증"""
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
            
        # DeepPCB 기본 구조 확인
        required_dirs = ['images', 'labels']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.warning(f"Creating missing directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_annotations(self, txt_path: str) -> List[Dict]:
        """DeepPCB 형식의 라벨 파일을 읽음"""
        annotations = []
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                    annotation = {
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    }
                    annotations.append(annotation)
                    
        except Exception as e:
            logger.error(f"Failed to load annotation {txt_path}: {e}")
            
        return annotations
    
    def convert_to_yolo_format(self, annotations: List[Dict], 
                             image_width: int, image_height: int) -> List[str]:
        """DeepPCB 라벨을 YOLO 형식으로 변환"""
        yolo_labels = []
        
        for ann in annotations:
            # DeepPCB는 이미 정규화된 좌표를 사용할 가능성이 높음
            # 필요시 좌표 변환 추가
            yolo_line = f"{ann['class_id']} {ann['x_center']} {ann['y_center']} {ann['width']} {ann['height']}"
            yolo_labels.append(yolo_line)
            
        return yolo_labels
    
    def prepare_dataset(self, output_path: str):
        """DeepPCB 데이터셋을 YOLO 학습용으로 준비"""
        output_path = Path(output_path)
        
        # 출력 디렉터리 생성
        (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        
        # PCBData 디렉터리 탐색
        pcb_data_path = self.dataset_path / 'PCBData'
        if pcb_data_path.exists():
            groups = [d for d in pcb_data_path.iterdir() if d.is_dir()]
            
            total_images = 0
            for group in groups:
                # 각 그룹의 이미지와 라벨 처리
                image_files = list(group.glob('*.jpg')) + list(group.glob('*.JPG'))
                
                for i, img_path in enumerate(image_files):
                    # 80% train, 20% val 분할
                    split = 'train' if i < len(image_files) * 0.8 else 'val'
                    
                    # 이미지 복사
                    dst_img = output_path / 'images' / split / img_path.name
                    if not dst_img.exists():
                        import shutil
                        shutil.copy2(img_path, dst_img)
                    
                    # 라벨 파일 처리
                    txt_path = img_path.with_suffix('.txt')
                    if txt_path.exists():
                        dst_txt = output_path / 'labels' / split / txt_path.name
                        shutil.copy2(txt_path, dst_txt)
                        
                    total_images += 1
                    
            logger.info(f"Prepared {total_images} images for training")
            
        # YAML 파일 생성
        self.create_yaml_config(output_path)
        
    def create_yaml_config(self, output_path: Path):
        """YOLO 학습용 YAML 설정 파일 생성"""
        yaml_content = f"""# DeepPCB Dataset Configuration
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: copper
  5: pin-hole

# Number of classes
nc: 6
"""
        
        yaml_path = output_path / 'deeppcb.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
            
        logger.info(f"Created YAML config: {yaml_path}")
        
    def get_class_names(self) -> List[str]:
        """DeepPCB 클래스 이름 반환"""
        return ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
