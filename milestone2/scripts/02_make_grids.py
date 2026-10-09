"""Generate the multi-object detection dataset in YOLO format.

Places 224x224 single-object photos side by side in 2x2 (448x448), 3x3 (672x672), 4x4 (896x896),
5x4 (1120x896) and 5x5 (1120x1120) grids, one fifth of the images each, with distinct objects in every
grid and flip and colour augmentation per photo. Only photos whose box passed the two-model check
(data/singles/boxes.csv) are used. Each split uses only its own source photos; a fingerprint check
proves no photo crosses splits, and every label is validated.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from detector.compose import (GRIDS, SPLITS, check_no_leakage, compose_split, load_objects, validate_labels,
                              write_data_yaml)
from detector.plots import draw_boxes, grid, read_yolo_labels

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--boxes", type=Path, default=ROOT / "data" / "singles" / "boxes.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "multi")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out
    shutil.rmtree(out, ignore_errors=True)
    objects = load_objects(args.boxes)
    classes = sorted(set().union(*(set(df.label) for df in objects.values())))
    rule = "=" * 70
    print(f"{rule}\nGENERATING GRID DATASET\n{rule}")
    print(f"Classes: {len(classes)} ({classes[0]} .. {classes[-1]})")
    print(f"Grids: " + ", ".join(f"{k} ({c * 224}x{r * 224}, {c * r} objects)" for k, (c, r) in GRIDS.items()))
    for s in SPLITS:
        print(f"  {s:<5}  {len(objects[s]):>5} labelled single-object photos, {objects[s].label.nunique()} objects")

    sizes = {"train": args.train, "val": args.val, "test": args.test}
    parts = []
    for s in SPLITS:
        parts.append(compose_split(s, objects[s], classes, sizes[s], out, args.seed))
        print(f"Generated {sizes[s]:>5} {s} images, {len(parts[-1]):>6} objects")
    manifest = pd.concat(parts)
    write_data_yaml(out, classes)
    manifest.to_csv(out / "manifest.csv", index=False)

    leakage = check_no_leakage(manifest)
    validation = validate_labels(out, len(classes))
    stats = {
        "classes": classes,
        "images": manifest.groupby("split").image.nunique().to_dict(),
        "objects": manifest.groupby("split").size().to_dict(),
        "grids": {s: g.drop_duplicates("image").grid.value_counts().sort_index().to_dict()
                  for s, g in manifest.groupby("split")},
        "min_instances_per_class": manifest.groupby(["split", "label"]).size().groupby("split").min().to_dict(),
        "leakage_check": leakage,
        "label_validation": validation,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    per_img = manifest.groupby("image").size()
    print(f"{rule}\nSUMMARY\n{rule}")
    print(f"Images: {len(per_img)}  ({', '.join(f'{s} {stats['images'][s]}' for s in SPLITS)})")
    print(f"Objects: {len(manifest)}  (4-25 per image, mean {per_img.mean():.2f})")
    print("Grids (train): " + ", ".join(f"{k} {v}" for k, v in stats["grids"]["train"].items()))
    print("Min. objects per class: " + ", ".join(f"{s} {stats['min_instances_per_class'][s]}" for s in SPLITS))
    print(f"Split check: {leakage['source_paths_in_2+_splits']} source paths and "
          f"{leakage['source_fingerprints_in_2+_splits']} source fingerprints in 2+ splits")
    print(f"Label check: {validation['images_checked']} images, {validation['boxes_checked']} boxes, "
          f"{validation['invalid']} invalid")

    first = manifest[manifest.split == "train"].drop_duplicates("image").groupby("grid").image.first()
    shots = []
    for g, stem in first.items():
        img = Image.open(out / "images" / "train" / f"{stem}.jpg")
        shots.append(draw_boxes(img, read_yolo_labels(out / "labels" / "train" / f"{stem}.txt", *img.size, classes)))
    (ROOT / "screenshots").mkdir(exist_ok=True)
    grid(shots, cols=len(shots), cell=640).save(ROOT / "screenshots" / "multi_object_samples.jpg", quality=90)
    print(f"Saved {out}/ (images, labels, data.yaml, manifest.csv), results/dataset_stats.json")


if __name__ == "__main__":
    main()
