"""Fine-tune COCO-pretrained YOLOv8 on the grid dataset (transfer learning).

Checkpoints go to --project (a Google Drive folder on Colab); if <project>/<name>/weights/last.pt
exists, training resumes from it.
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
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "multi" / "data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1120)
    ap.add_argument("--project", type=Path, default=ROOT / "runs")
    ap.add_argument("--name", default=None, help="run name under the project folder (default: model stem)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    name = args.name or Path(args.model).stem
    last = args.project / name / "weights" / "last.pt"
    if last.exists():
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        weights = ROOT / args.model
        YOLO(str(weights) if weights.exists() else args.model).train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
            project=str(args.project), name=name, exist_ok=True, save_period=5,
            patience=15, seed=42, deterministic=False, workers=args.workers, plots=True,
        )
    best = args.project / name / "weights" / "best.pt"
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "models" / f"{name}_best.pt")
    print(f"-> models/{name}_best.pt")


if __name__ == "__main__":
    main()
