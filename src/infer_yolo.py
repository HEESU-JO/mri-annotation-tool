import os, argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)          # 학습된 pt
    ap.add_argument("--source", default="data/yolo/images/val")
    ap.add_argument("--project", default="reports")
    ap.add_argument("--name", default="infer")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--save_conf", action="store_true")
    ap.add_argument("--save_crop", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        exist_ok=True
    )

if __name__ == "__main__":
    main()