# 🏃 Process View

## 학습 파이프라인

```
yolo_train.py
 ├── load config
 ├── download or load pretrained model
 ├── remove labels.cache if exists
 ├── load and augment dataset
 ├── train model
 └── save to models/mymodel_v1.pt
```

## 추론 파이프라인

```
infer.py
 ├── load config
 ├── load trained model
 ├── load and preprocess test images
 ├── run inference (GPU by default)
 └── summarize and visualize results
```

- 학습과 추론은 명확히 분리되어 있어 운영 시에는 `yolo_train.py`를 포함할 필요가 없습니다.
