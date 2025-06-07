# Semiconductor AOI Inspector

기본적인 CLI 기반 반도체 자동 광학 검사(AOI) 도구의 골격입니다. `AOI_Inspector_SDD.md` 문서에서 정의된 모듈 구조를 참고하여 구성되었습니다.

## 구조
- `main.py`: 프로그램 진입점
- `modules/`: 데이터 로딩, 전처리, 추론, 후처리, 시각화 모듈
- `models/`: 사전 학습된 모델 파일 보관 위치
- `config.yaml`: 실행 설정 파일

## 실행 방법
```
python main.py --config config.yaml
```

현재는 예제 수준의 코드로, 실제 모델과 데이터는 별도 준비가 필요합니다.
