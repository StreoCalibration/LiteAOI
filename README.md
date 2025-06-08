# LiteAOI

이 저장소는 반도체 제조 공정에서 결함을 자동으로 검출하기 위한 "Semiconductor AOI Inspector" 프로젝트의 초기 구조입니다.

## Visual Studio Code 환경 설정

1. Python 3.10 이상이 설치되어 있는지 확인합니다.
2. 이 저장소를 클론한 후 VSCode에서 `File > Open Folder...` 메뉴를 통해 폴더를 엽니다.
3. VSCode에서 Python 확장(`ms-python.python`)을 설치합니다.
4. 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt

```

5. `main.py`와 모듈들을 추가하여 개발을 진행합니다.

## 요구 사항

프로젝트 문서에 명시된 바와 같이, 이 도구는 Python 3.x 환경에서 PyTorch, OpenCV, YOLOv5/YOLOv8, Anomalib, Albumentations 등을 활용합니다. 주요 기술 스택은 아래와 같습니다.

- Python 3.10 이상
- PyTorch, Ultralytics YOLO, Anomalib
- OpenCV, NumPy
- Albumentations

이 패키지들은 `requirements.txt`에 정리되어 있습니다. 아래 명령어로 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```
