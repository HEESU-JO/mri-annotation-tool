import os, random, argparse
import numpy as np
from ultralytics import YOLO

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="reports")     # 결과 폴더
    ap.add_argument("--name", default="train_yolov8n") # 런 이름
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    model = YOLO(args.model)

    # 학습
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    # 검증
    metrics = model.val(project=args.project, name=f"{args.name}_val", exist_ok=True)
    print(f"mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
