"""Train SimpleCNN, ResNet18, MobileNetV3 and EfficientNetB0 on data/{train,val,test}.

    python scripts/03_train_models.py                              # all models
    python scripts/03_train_models.py --models ResNet18 --epochs 10

Writes models/<Model>.pt and results/<Model>_{classification_report.json, training_history.png, confusion_matrix.png}.
"""
import argparse
import sys
from pathlib import Path

from classifier.data import build_loaders, load_split, pick_device
from classifier.models import MODEL_NAMES
from classifier.train import train_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int, help="override the per-model default")
    ap.add_argument("--lr", type=float, help="override the per-model default")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0 if sys.platform == "darwin" else 4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = pick_device()
    split = load_split(ROOT / "data")
    loaders, classes = build_loaders(split, args.batch_size, args.workers)
    print(f"{len(classes)} classes |", split.groupby("split").size().to_dict())
    for name in args.models:
        train_model(name, loaders, classes, device, ROOT / "models", ROOT / "results",
                    args.epochs, args.lr, args.patience, args.seed)


if __name__ == "__main__":
    main()
