"""Identify the object in one or more photos.

    python scripts/05_predict.py photo.jpg [more.jpg ...]
    python scripts/05_predict.py photo.jpg --model models/MobileNetV3.pt
"""
import argparse
from pathlib import Path

from classifier.data import pick_device
from classifier.predict import identify, load_model

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=ROOT / "models" / "EfficientNetB0.pt")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    device = pick_device()
    model, classes = load_model(args.model, device)
    for path in args.images:
        preds = identify(model, classes, path, device, args.k)
        print(f"{path}: {preds[0][0]} ({preds[0][1]:.1%})  top-{args.k}: "
              + ", ".join(f"{c} {p:.1%}" for c, p in preds))


if __name__ == "__main__":
    main()
