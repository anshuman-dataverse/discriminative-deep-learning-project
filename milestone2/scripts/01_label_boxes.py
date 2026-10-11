"""Create a bounding box for every single-object photo with two open-vocabulary detectors and cross-check them.

Grounding DINO and OWLv2 each get a short text prompt per Object ID (detector.names) and return their
highest-scoring box. A photo is kept when both models find the object and their boxes overlap with
IoU >= --min-iou; the label is the mean of the two boxes. Photos where a model finds nothing, the two
boxes disagree, or the object covers less than --min-area of the photo are left out and counted.
Progress is saved to --partial every 50 batches and picked up again on the next run. Saved photos are
decided again from their stored boxes; a photo is run through the detectors again only when its prompt
changed or a detector found nothing at a higher score threshold than the current one.
Writes data/singles/boxes.csv, results/label_stats.json and screenshots/label_check_{kept,dropped}.jpg.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from detector.boxes import GDINO_THRESHOLD, OWL_THRESHOLD, GroundingDino, Owl, device
from detector.compose import SPLITS, TILE
from detector.evaluate import iou
from detector.names import PROMPTS
from detector.plots import caption, draw_boxes, grid

ROOT = Path(__file__).resolve().parent.parent
SINGLES = ROOT / "data" / "singles"
DRIVE = Path("/content/drive/MyDrive/IE7615_Milestone2")


def decide(g, o, min_iou: float, min_area: float):
    if g is None and o is None:
        return "no_box", None, None
    if g is None or o is None:
        return "one_model", None, None
    v = iou(g[1], o[1])
    if v < min_iou:
        return "disagree", v, None
    box = [min(max((a + b) / 2, 0), TILE) for a, b in zip(g[1], o[1])]
    if (box[2] - box[0]) * (box[3] - box[1]) < min_area * TILE * TILE:
        return "too_small", v, box
    return "kept", v, box


def stale(r) -> bool:
    """Saved row that must be labelled again: new prompt, or a detector found nothing at a stricter threshold."""
    return (r["prompt"] != PROMPTS.get(r["label"])
            or (pd.isna(r["gx1"]) and _thr(r, "gdino_thr", 0.2) > GDINO_THRESHOLD)
            or (pd.isna(r["ox1"]) and _thr(r, "owl_thr", 0.1) > OWL_THRESHOLD))


def _thr(r, key: str, first_run: float) -> float:
    v = r.get(key)
    return first_run if v is None or pd.isna(v) else v


def redecide(r: pd.Series, min_iou: float, min_area: float) -> pd.Series:
    g = None if pd.isna(r.gx1) else (r.gdino_score, (r.gx1, r.gy1, r.gx2, r.gy2))
    o = None if pd.isna(r.ox1) else (r.owl_score, (r.ox1, r.oy1, r.ox2, r.oy2))
    status, v, box = decide(g, o, min_iou, min_area)
    r = r.copy()
    r["status"], r["iou"] = status, None if v is None else round(v, 3)
    for i, k in enumerate(("x1", "y1", "x2", "y2")):
        r[k] = None if box is None else round(box[i], 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", type=Path, default=SINGLES)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.015)
    ap.add_argument("--partial", type=Path, default=(DRIVE if DRIVE.exists() else SINGLES) / "boxes_partial.csv")
    args = ap.parse_args()

    photos = pd.DataFrame([{"path": str(f), "label": d.name, "split": s}
                           for s in SPLITS for d in sorted((args.singles / s).iterdir()) if d.is_dir()
                           for f in sorted(d.glob("*.jpg"))])
    missing = sorted(set(photos.label) - set(PROMPTS))
    if missing:
        raise SystemExit(f"no text prompt for {missing}")
    dev = device()
    gdino, owl = GroundingDino(dev), Owl(dev)
    rule = "=" * 70
    print(f"{rule}\nLABELLING {len(photos)} PHOTOS ({photos.label.nunique()} objects) ON {dev}\n{rule}", flush=True)

    rows = [r for r in (pd.read_csv(args.partial).to_dict("records") if args.partial.exists() else [])
            if not stale(r)]
    todo = photos[~photos.path.isin({r["path"] for r in rows})]
    if rows:
        print(f"resuming: {len(rows)} photos already labelled in {args.partial}, {len(todo)} to go", flush=True)
    for start in range(0, len(todo), args.batch):
        chunk = todo.iloc[start:start + args.batch]
        images = [Image.open(p).convert("RGB") for p in chunk.path]
        prompts = [PROMPTS[l] for l in chunk.label]
        for r, g, o in zip(chunk.itertuples(), gdino(images, prompts), owl(images, prompts)):
            status, v, box = decide(g, o, args.min_iou, args.min_area)
            rows.append({"path": r.path, "label": r.label, "split": r.split, "prompt": PROMPTS[r.label],
                         "gdino_thr": GDINO_THRESHOLD, "owl_thr": OWL_THRESHOLD, "status": status, "iou": None if v is None else round(v, 3),
                         "gdino_score": g and round(g[0], 3), "owl_score": o and round(o[0], 3),
                         **{f"g{k}": g and g[1][i] for i, k in enumerate(("x1", "y1", "x2", "y2"))},
                         **{f"o{k}": o and o[1][i] for i, k in enumerate(("x1", "y1", "x2", "y2"))},
                         **{k: box and round(box[i], 1) for i, k in enumerate(("x1", "y1", "x2", "y2"))}})
        if (start // args.batch) % 50 == 49 or start + len(chunk) == len(todo):
            pd.DataFrame(rows).to_csv(args.partial, index=False)
            print(f"  {len(rows):>5} / {len(photos)}", flush=True)

    df = pd.DataFrame(rows)
    df = df.apply(lambda r: redecide(r, args.min_iou, args.min_area), axis=1)
    df.to_csv(args.singles / "boxes.csv", index=False)
    kept = df[df.status == "kept"]
    per_class = kept.groupby(["split", "label"]).size().unstack(0).reindex(sorted(df.label.unique())).fillna(0)
    stats = {
        "photos": len(df),
        "kept": len(kept),
        "left_out": df[df.status != "kept"].status.value_counts().to_dict(),
        "kept_by_split": kept.groupby("split").size().to_dict(),
        "left_out_by_class": df[df.status != "kept"].groupby("label").size().sort_values(ascending=False).head(10).to_dict(),
        "min_kept_per_class": per_class.min().astype(int).to_dict(),
        "classes_without_test_photos": per_class.index[per_class.get("test", 0) == 0].tolist(),
        "median_iou_kept": round(float(kept.iou.median()), 3),
        "mean_box_area_fraction": round(float(((kept.x2 - kept.x1) * (kept.y2 - kept.y1)).mean() / TILE ** 2), 3),
        "min_iou": args.min_iou, "min_area": args.min_area,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "label_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{rule}\nSUMMARY\n{rule}")
    print(f"Kept           {len(kept):>5} / {len(df)}  ({len(kept) / len(df):.1%})")
    for k, v in stats["left_out"].items():
        print(f"Left out       {v:>5}  {k}")
    print("Kept by split  " + ", ".join(f"{s} {stats['kept_by_split'].get(s, 0)}" for s in SPLITS))
    print("Min per class  " + ", ".join(f"{s} {stats['min_kept_per_class'].get(s, 0)}" for s in SPLITS))
    print(f"Median IoU between the two models (kept): {stats['median_iou_kept']}")
    if stats["classes_without_test_photos"]:
        print(f"WARNING: no kept test photos for {stats['classes_without_test_photos']}")

    rng = random.Random(0)
    (ROOT / "screenshots").mkdir(exist_ok=True)
    for tag, sel in (("kept", kept), ("dropped", df[df.status != "kept"])):
        sample = sel.sample(min(24, len(sel)), random_state=rng.randint(0, 9999)) if len(sel) else sel
        tiles = []
        for r in sample.itertuples():
            boxes = []
            if pd.notna(r.gx1):
                boxes.append((*(2 * v for v in (r.gx1, r.gy1, r.gx2, r.gy2)), "DINO", 0))
            if pd.notna(r.ox1):
                boxes.append((*(2 * v for v in (r.ox1, r.oy1, r.ox2, r.oy2)), "OWL", 3))
            img = draw_boxes(Image.open(r.path).convert("RGB").resize((2 * TILE, 2 * TILE)), boxes)
            caption(img, f"{r.label} {r.prompt}: {r.status}" + ("" if pd.isna(r.iou) else f" IoU {r.iou:.2f}"))
            tiles.append(img)
        if tiles:
            grid(tiles, cols=6, cell=300).save(ROOT / "screenshots" / f"label_check_{tag}.jpg", quality=90)
    print(f"Saved data/singles/boxes.csv, results/label_stats.json, screenshots/label_check_{{kept,dropped}}.jpg")


if __name__ == "__main__":
    main()
