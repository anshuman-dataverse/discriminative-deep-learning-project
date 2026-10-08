"""Generate the multi-object detection dataset in YOLO format.

Pastes 2-6 single-object photos (data/singles/<split>) onto a background-only photo in one of three
layouts: scattered with gaps (50%), a 2x2 / 3x3 concatenation grid (25%), or a packed collage where
objects touch or overlap slightly (25%), with flip and colour augmentation. Each split uses only its
own source images; a fingerprint check proves no image crosses splits, and every label is validated.

    python scripts/01_make_multi_object.py                     # 1200 / 250 / 250 images
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from detector.compose import (SPLITS, check_no_leakage, compose_split, load_objects, split_backgrounds,
                              validate_labels, write_data_yaml)
from detector.plots import draw_boxes, grid, read_yolo_labels

ROOT = Path(__file__).resolve().parent.parent
SINGLES = ROOT / "data" / "singles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=250)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--singles", type=Path, default=SINGLES, help="folder with train/val/test/<OBJ###>/")
    ap.add_argument("--backgrounds", type=Path, default=SINGLES / "backgrounds", help="background-only photos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = ROOT / "data" / "multi"
    shutil.rmtree(out, ignore_errors=True)
    objects = load_objects(args.singles)
    classes = sorted(set().union(*(set(df.label) for df in objects.values())))
    backgrounds = split_backgrounds(args.backgrounds, seed=args.seed)
    rule = "=" * 70
    print(f"{rule}\nGENERATING MULTI-OBJECT DATASET\n{rule}")
    print(f"Classes: {len(classes)} ({classes[0]} .. {classes[-1]})")
    for s in SPLITS:
        print(f"  {s:<5}  {len(objects[s]):>5} single-object photos  {len(backgrounds[s]):>4} backgrounds")

    sizes = {"train": args.train, "val": args.val, "test": args.test}
    parts = []
    for s in SPLITS:
        parts.append(compose_split(s, objects[s], backgrounds[s], classes, sizes[s], out, args.seed))
        print(f"Generated {sizes[s]:>5} {s} images, {len(parts[-1]):>5} objects")
    manifest = pd.concat(parts)
    write_data_yaml(out, classes)
    manifest.to_csv(out / "manifest.csv", index=False)

    leakage = check_no_leakage(manifest)
    validation = validate_labels(out, len(classes))
    stats = {
        "classes": classes,
        "images": manifest.groupby("split").image.nunique().to_dict(),
        "objects": manifest.groupby("split").size().to_dict(),
        "objects_per_image": manifest.groupby("image").size().describe().round(2).to_dict(),
        "layouts": manifest.drop_duplicates("image").layout.value_counts().to_dict(),
        "min_instances_per_class": manifest.groupby(["split", "label"]).size().groupby("split").min().to_dict(),
        "leakage_check": leakage,
        "label_validation": validation,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    per_img = manifest.groupby("image").size()
    print(f"{rule}\nSUMMARY\n{rule}")
    print(f"Images: {len(per_img)}  ({', '.join(f'{s} {stats['images'][s]}' for s in SPLITS)})")
    print(f"Objects: {len(manifest)}  (2-6 per image, mean {per_img.mean():.2f})")
    print("Layouts: " + ", ".join(f"{k} {v}" for k, v in stats["layouts"].items()))
    print("Min. objects per class: " + ", ".join(f"{s} {stats['min_instances_per_class'][s]}" for s in SPLITS))
    print(f"Split check: {leakage['source_paths_in_2+_splits']} source paths and "
          f"{leakage['source_fingerprints_in_2+_splits']} source fingerprints in 2+ splits; "
          f"{leakage['background_fingerprints_in_2+_splits']} shared backgrounds")
    print(f"Label check: {validation['images_checked']} images, {validation['boxes_checked']} boxes, "
          f"{validation['invalid']} invalid")

    first = manifest[manifest.split == "train"].drop_duplicates("image").groupby("layout").image.apply(list)
    samples = [out / "images" / "train" / f"{stem}.jpg"
               for layout, n in (("scatter", 3), ("grid", 2), ("collage", 3)) for stem in first[layout][:n]]
    shots = [draw_boxes(Image.open(f), read_yolo_labels(out / "labels" / "train" / f"{f.stem}.txt", 640, classes))
             for f in samples]
    (ROOT / "screenshots").mkdir(exist_ok=True)
    grid(shots, cols=4).save(ROOT / "screenshots" / "multi_object_samples.jpg", quality=90)
    print(f"Saved data/multi/ (images, labels, data.yaml, manifest.csv), results/dataset_stats.json")


if __name__ == "__main__":
    main()
