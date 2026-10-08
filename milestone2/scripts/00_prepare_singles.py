"""Clean and split the full shared-Drive download (final Object IDs) into data/singles/{train,val,test}/OBJ###/.

Reuses the milestone 1 pipeline: 224x224 RGB with EXIF orientation, near-duplicates removed,
stratified 70/15/15 split. Background-only photos are kept in data/singles/backgrounds/ for compositing.

    python scripts/00_prepare_singles.py                  # reads data/raw/*/
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "milestone1" / "scripts"))

from classifier.data import audit, make_split, normalize, remove_duplicates, write_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="*", help="extracted Drive download folders (default: data/raw/*)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = args.raw or sorted(p for p in (ROOT / "data" / "raw").iterdir() if p.is_dir())
    out = ROOT / "data" / "singles"

    report = audit(raw)
    print(f"{len(report)} folders, {report.images.sum()} images, {report.not_224x224.sum()} not 224x224, "
          f"{(~report.name_ok).sum()} misnamed")

    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp)
        summary = normalize(raw, clean)
        print(f"{len(summary)} objects, {summary.object_imgs.sum()} object + {summary.background_imgs.sum()} background images")
        removed, cross = remove_duplicates(clean)
        print(f"removed {len(removed)} duplicates ({cross} groups spanning two objects)")
        split = write_split(make_split(clean, seed=args.seed), out)

        bg_dir = out / "backgrounds"
        shutil.rmtree(bg_dir, ignore_errors=True)
        bg_dir.mkdir(parents=True)
        for f in clean.rglob("*_bg_*.jpg"):
            shutil.copy2(f, bg_dir / f.name)

    classes = sorted(split.label.unique())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "class_labels.json").write_text(json.dumps(classes, indent=2))
    print(f"{len(classes)} classes |", split.groupby("split").size().to_dict(),
          f"| {len(list(bg_dir.glob('*.jpg')))} backgrounds -> data/singles/")


if __name__ == "__main__":
    main()
