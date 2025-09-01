# MRI Lesion Detection (YOLOv8)
MRI 병변을 YOLOv8으로 탐지하는 프로젝트입니다.  
데이터 준비 → 시각화 검증 → 학습 → 추론까지 **재현 가능**하도록 최소 구성을 제공합니다.

## 1) 구조
```
mri-annotation-yolo/
├── src/                    # 학습(train), 추론(inference) 스크립트
│   ├── train_yolo.py
│   └── infer_yolo.py
├── tools/                  # 데이터 검증 및 시각화 도구
│   ├── import_and_check.py
│   └── visualize_yolo_samples.py
├── data/                   # (비공개) YOLO 형식 데이터셋 구조
│   └── yolo/
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── weights/                # (비공개) 학습된 모델 가중치
├── reports/                # (비공개) 학습 로그 및 추론 결과
├── requirements.txt        # 실행에 필요한 패키지 목록
├── .gitignore              # 민감 데이터 제외 설정
└── README.md               # 프로젝트 설명 파일
```

> ⚠️ `data/`, `weights/`, `reports/` 폴더는 `.gitignore`로 보호되어 있어 GitHub에는 올라가지 않습니다.  
> 🔍 구조 확인을 위한 `data.yaml` 파일만 예외적으로 포함되어 있습니다.
