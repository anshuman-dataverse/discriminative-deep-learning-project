"""Clean the raw download and write the stratified 70/15/15 split to data/{train,val,test}/OBJ###/.

Every image is resized to 224x224 RGB with EXIF orientation applied, renamed OBJ###_###.jpg,
and near-duplicates are removed so no photo appears in both train and test.
Background-only photos are excluded.
"""
import argparse
import json
import tempfile
from pathlib import Path

from classifier.data import make_split, normalize, remove_duplicates, write_split

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="*", help="extracted Drive download folders (default: data/raw/*)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = args.raw or sorted(p for p in (ROOT / "data" / "raw").iterdir() if p.is_dir())

    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp)
        summary = normalize(raw, clean)
        print(f"{len(summary)} objects, {summary.object_imgs.sum()} object + {summary.background_imgs.sum()} background images")
        removed, cross = remove_duplicates(clean)
        print(f"removed {len(removed)} duplicates ({cross} groups spanning two objects)")
        split = write_split(make_split(clean, seed=args.seed), ROOT / "data")

    classes = sorted(split.label.unique())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "class_labels.json").write_text(json.dumps(classes, indent=2))
    print(f"{len(classes)} classes |", split.groupby("split").size().to_dict(), "-> data/{train,val,test}/")


if __name__ == "__main__":
    main()
