"""Build multi-object grid images by placing 224x224 single-object photos side by side, with YOLO labels.

Each split draws only on its own source photos: train grids use data/singles/train, val uses val,
test uses test, so no source image appears in more than one split. Each label is the photo's
object box (from detector.boxes) shifted to the photo's cell.
"""
import hashlib
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

SPLITS = ("train", "val", "test")
TILE = 224
GRIDS = {"2x2": (2, 2), "3x3": (3, 3), "4x4": (4, 4), "5x4": (5, 4), "5x5": (5, 5)}


def fingerprint(path) -> str:
    """16x16 grayscale hash, the same near-duplicate fingerprint milestone 1 used."""
    with Image.open(path) as im:
        return hashlib.md5(ImageOps.exif_transpose(im).convert("L").resize((16, 16)).tobytes()).hexdigest()


def load_objects(boxes_csv: Path, singles: Path) -> dict[str, pd.DataFrame]:
    """Kept rows of the box table -> {split: DataFrame(path, label, x1, y1, x2, y2)}, with paths under singles/."""
    df = pd.read_csv(boxes_csv)
    df = df[df.status == "kept"].copy()
    df["path"] = [str(singles / s / l / Path(p).name) for s, l, p in zip(df.split, df.label, df.path)]
    return {s: df[df.split == s].reset_index(drop=True) for s in SPLITS}


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(0.75, 1.25))
    return img


class ObjectSampler:
    """Cycles through a shuffled pool so every source photo is used about equally often."""

    def __init__(self, df: pd.DataFrame, rng: random.Random):
        self.rng = rng
        self.by_label = {lbl: g.to_dict("records") for lbl, g in df.groupby("label")}
        self.queues = {lbl: [] for lbl in self.by_label}
        self.label_queue = []

    def _next_label(self):
        if not self.label_queue:
            self.label_queue = list(self.by_label)
            self.rng.shuffle(self.label_queue)
        return self.label_queue.pop()

    def draw(self, n: int) -> list[dict]:
        """n photo records with distinct labels."""
        labels = []
        while len(labels) < n:
            lbl = self._next_label()
            if lbl not in labels:
                labels.append(lbl)
        picks = []
        for lbl in labels:
            if not self.queues[lbl]:
                self.queues[lbl] = self.by_label[lbl][:]
                self.rng.shuffle(self.queues[lbl])
            picks.append(self.queues[lbl].pop())
        return picks


def compose_split(split: str, objects: pd.DataFrame, classes: list[str], n_images: int, out: Path,
                  seed: int) -> pd.DataFrame:
    """Write out/images/<split>/*.jpg and out/labels/<split>/*.txt; return one row per placed object.

    Grid sizes cycle through GRIDS so each size gets the same share of images."""
    rng = random.Random(f"{seed}-{split}")
    sampler = ObjectSampler(objects, rng)
    cls_idx = {c: i for i, c in enumerate(classes)}
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    grid_names = list(GRIDS)
    rows = []
    for i in range(n_images):
        name = f"{split}_{i:04d}"
        grid = grid_names[i % len(grid_names)]
        cols, nrows = GRIDS[grid]
        W, H = cols * TILE, nrows * TILE
        canvas = Image.new("RGB", (W, H))
        lines = []
        for k, rec in enumerate(sampler.draw(cols * nrows)):
            ox, oy = (k % cols) * TILE, (k // cols) * TILE
            tile = Image.open(rec["path"]).convert("RGB")
            x1, y1, x2, y2 = rec["x1"], rec["y1"], rec["x2"], rec["y2"]
            if rng.random() < 0.5:
                tile = ImageOps.mirror(tile)
                x1, x2 = TILE - x2, TILE - x1
            canvas.paste(_augment(tile, rng), (ox, oy))
            x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
            lines.append(f"{cls_idx[rec['label']]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            rows.append({"image": name, "split": split, "grid": grid, "width": W, "height": H,
                         "source": rec["path"], "label": rec["label"],
                         "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)})
        canvas.save(out / "images" / split / f"{name}.jpg", "JPEG", quality=92)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def check_no_leakage(manifest: pd.DataFrame) -> dict:
    """Assert no source photo (by path or by near-duplicate fingerprint) is shared by two splits."""
    report = {}
    used = manifest[["split", "source"]].drop_duplicates()
    used = used.assign(fp=used.source.map(fingerprint))
    for key, tag in (("source", "paths"), ("fp", "fingerprints")):
        per = used.groupby(key).split.nunique()
        shared = per[per > 1]
        report[f"source_{tag}_in_2+_splits"] = int(len(shared))
        if len(shared):
            raise AssertionError(f"{len(shared)} source photos appear in more than one split: {list(shared.index[:5])}")
    report["unique_source_images"] = used.groupby("split").source.nunique().to_dict()
    return report


def validate_labels(out: Path, n_classes: int) -> dict:
    """Check every image/label pair: matching files, a grid-sized image, one box per cell, 5 values per line,
    class index in range, box inside the image with positive size. Raises on the first problem."""
    sizes = {(c * TILE, r * TILE): c * r for c, r in GRIDS.values()}
    checked = boxes = 0
    for split in SPLITS:
        images = {f.stem for f in (out / "images" / split).glob("*.jpg")}
        labels = {f.stem for f in (out / "labels" / split).glob("*.txt")}
        if images != labels:
            raise AssertionError(f"{split}: {len(images ^ labels)} images without labels or labels without images")
        for stem in sorted(labels):
            with Image.open(out / "images" / split / f"{stem}.jpg") as im:
                size = im.size
            if size not in sizes:
                raise AssertionError(f"{split}/{stem}.jpg is {size}, not a grid size")
            lines = [l.split() for l in (out / "labels" / split / f"{stem}.txt").read_text().split("\n") if l.strip()]
            if len(lines) != sizes[size]:
                raise AssertionError(f"{split}/{stem}.txt has {len(lines)} objects, expected {sizes[size]}")
            for v in lines:
                bad = len(v) != 5
                if not bad:
                    c, (xc, yc, w, h) = int(v[0]), map(float, v[1:])
                    eps = 1e-6
                    bad = (not 0 <= c < n_classes or not (0 < w <= 1 and 0 < h <= 1)
                           or min(xc - w / 2, yc - h / 2) < -eps or max(xc + w / 2, yc + h / 2) > 1 + eps)
                if bad:
                    raise AssertionError(f"{split}/{stem}.txt: invalid line {' '.join(v)}")
            checked += 1
            boxes += len(lines)
    return {"images_checked": checked, "boxes_checked": boxes, "invalid": 0}


def write_data_yaml(out: Path, classes: list[str]):
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames:\n{names}\n")
