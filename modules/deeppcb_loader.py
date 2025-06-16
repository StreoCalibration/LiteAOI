"""DeepPCB 데이터셋 로더 모듈"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import shutil

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
            
        # DeepPCB의 PCBData 폴더 확인
        pcb_data_path = self.dataset_path / 'PCBData'
        if not pcb_data_path.exists():
            raise ValueError(f"PCBData directory not found in {self.dataset_path}")
    
    def load_deeppcb_annotation(self, txt_path: str) -> List[Dict]:
        """DeepPCB 형식의 라벨 파일을 읽음
        
        DeepPCB 형식: x1 y1 x2 y2 class_id
        - (x1, y1): 좌상단 좌표
        - (x2, y2): 우하단 좌표
        - class_id: 1-6 (1:open, 2:short, 3:mousebite, 4:spur, 5:copper, 6:pin-hole)
        """
        annotations = []
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # x1, y1, x2, y2, class_id
                    annotation = {
                        'x1': int(parts[0]),
                        'y1': int(parts[1]),
                        'x2': int(parts[2]),
                        'y2': int(parts[3]),
                        'class_id': int(parts[4]) - 1  # DeepPCB는 1-6, YOLO는 0-5
                    }
                    annotations.append(annotation)
                    
        except Exception as e:
            logger.error(f"Failed to load annotation {txt_path}: {e}")
            
        return annotations
    
    def convert_to_yolo_format(self, annotations: List[Dict], 
                             image_width: int, image_height: int) -> List[str]:
        """DeepPCB 라벨을 YOLO 형식으로 변환
        
        YOLO 형식: class_id x_center y_center width height (모두 정규화된 값)
        """
        yolo_labels = []
        
        for ann in annotations:
            # 바운딩 박스 좌표를 중심점과 크기로 변환
            x_center = (ann['x1'] + ann['x2']) / 2.0 / image_width
            y_center = (ann['y1'] + ann['y2']) / 2.0 / image_height
            width = (ann['x2'] - ann['x1']) / image_width
            height = (ann['y2'] - ann['y1']) / image_height
            
            # YOLO 형식으로 포맷
            yolo_line = f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
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
        
        # DeepPCB 구조 탐색
        pcb_data_path = self.dataset_path / 'PCBData'
        total_images = 0
        processed_images = 0
        
        # 모든 그룹 폴더 탐색
        group_folders = sorted([d for d in pcb_data_path.iterdir() if d.is_dir() and d.name.startswith('group')])
        
        if not group_folders:
            raise ValueError(f"No group folders found in {pcb_data_path}")
        
        logger.info(f"Found {len(group_folders)} group folders")
        
        # 각 그룹 폴더 처리
        for group_idx, group_folder in enumerate(group_folders):
            # 그룹 내의 하위 폴더 찾기
            sub_folders = [d for d in group_folder.iterdir() if d.is_dir() and not d.name.endswith('_not')]
            
            for sub_folder in sub_folders:
                # 이미지 파일들 찾기
                image_files = list(sub_folder.glob('*.jpg')) + list(sub_folder.glob('*.JPG'))
                
                # 대응하는 라벨 폴더
                label_folder = group_folder / f"{sub_folder.name}_not"
                
                if not label_folder.exists():
                    logger.warning(f"Label folder not found: {label_folder}")
                    continue
                
                for img_path in image_files:
                    total_images += 1
                    
                    # 대응하는 라벨 파일
                    label_path = label_folder / f"{img_path.stem}.txt"
                    
                    if not label_path.exists():
                        logger.warning(f"Label file not found: {label_path}")
                        continue
                    
                    # 80% train, 20% val 분할 (그룹 단위로)
                    split = 'train' if group_idx < len(group_folders) * 0.8 else 'val'
                    
                    # 이미지 읽어서 크기 확인
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.error(f"Failed to read image: {img_path}")
                        continue
                    
                    height, width = img.shape[:2]
                    
                    # 라벨 로드 및 변환
                    annotations = self.load_deeppcb_annotation(str(label_path))
                    yolo_labels = self.convert_to_yolo_format(annotations, width, height)
                    
                    if not yolo_labels:
                        logger.warning(f"No valid labels for: {img_path}")
                        continue
                    
                    # 파일 복사
                    dst_img = output_path / 'images' / split / f"{img_path.stem}.jpg"
                    dst_label = output_path / 'labels' / split / f"{img_path.stem}.txt"
                    
                    # 이미지 복사
                    shutil.copy2(img_path, dst_img)
                    
                    # YOLO 형식 라벨 저장
                    with open(dst_label, 'w') as f:
                        f.write('\n'.join(yolo_labels))
                    
                    processed_images += 1
                    
                    if processed_images % 100 == 0:
                        logger.info(f"Processed {processed_images}/{total_images} images")
        
        logger.info(f"Processing complete!")
        logger.info(f"Total images found: {total_images}")
        logger.info(f"Successfully processed: {processed_images}")
        
        train_count = len(list((output_path / 'images' / 'train').glob('*.jpg')))
        val_count = len(list((output_path / 'images' / 'val').glob('*.jpg')))
        logger.info(f"Train: {train_count} images")
        logger.info(f"Val: {val_count} images")
        
        # YAML 파일 생성
        self.create_yaml_config(output_path)
        
    def create_yaml_config(self, output_path: Path):
        """YOLO 학습용 YAML 설정 파일 생성"""
        yaml_content = f"""# DeepPCB Dataset Configuration
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes (DeepPCB의 1-6을 0-5로 변환)
names:
  0: open        # 단선
  1: short       # 단락
  2: mousebite   # 마우스바이트
  3: spur        # 스퍼
  4: copper      # 구리잔여물
  5: pin-hole    # 핀홀

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
