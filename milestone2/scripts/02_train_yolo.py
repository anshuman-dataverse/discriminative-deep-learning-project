"""Fine-tune COCO-pretrained YOLOv8 on the multi-object dataset (transfer learning).
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", default=None, help="run name under runs/ (default: model stem)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    weights = ROOT / args.model
    model = YOLO(str(weights) if weights.exists() else args.model)
    name = args.name or Path(args.model).stem
    model.train(
        data=str(ROOT / "data" / "multi" / "data.yaml"),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=str(ROOT / "runs"), name=name, exist_ok=True,
        patience=15, seed=42, deterministic=False, workers=4, plots=True,
    )
    best = ROOT / "runs" / name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / f"{name}_best.pt")
    print(f"-> models/{name}_best.pt")


if __name__ == "__main__":
    main()
