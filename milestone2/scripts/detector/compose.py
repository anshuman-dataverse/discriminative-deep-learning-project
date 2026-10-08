"""Build multi-object images by pasting single-object photos onto background photos, with YOLO labels.

Each split draws only on its own source images: train composites use data/singles/train,
val uses val, test uses test, and the background photos are split the same way, so no source
image appears in more than one split.
"""
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

Image.MAX_IMAGE_PIXELS = None

SPLITS = ("train", "val", "test")
CANVAS = 640
EXTS = {".jpg", ".jpeg", ".png"}


def fingerprint(path) -> str:
    """16x16 grayscale hash, the same near-duplicate fingerprint milestone 1 used."""
    with Image.open(path) as im:
        return hashlib.md5(ImageOps.exif_transpose(im).convert("L").resize((16, 16)).tobytes()).hexdigest()


def load_objects(single_dir: Path) -> dict[str, pd.DataFrame]:
    """milestone1/data/<split>/<OBJ###>/*.jpg -> {split: DataFrame(path, label)}."""
    out = {}
    for split in SPLITS:
        rows = [{"path": str(f), "label": d.name}
                for d in sorted((single_dir / split).iterdir()) if d.is_dir()
                for f in sorted(d.glob("*.jpg"))]
        if not rows:
            raise SystemExit(f"no images under {single_dir / split}; run scripts/00_prepare_singles.py first")
        out[split] = pd.DataFrame(rows)
    return out


def split_backgrounds(bg_dir: Path, val: float = 0.15, test: float = 0.15, seed: int = 42) -> dict[str, list[Path]]:
    """Background-only photos (*bg* in the name), split 70/15/15; near-duplicates always share a split."""
    groups = defaultdict(list)
    for f in sorted(bg_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in EXTS and "bg" in f.stem.lower() and not f.name.startswith("."):
            groups[fingerprint(f)].append(f)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test, n_val = round(len(keys) * test), round(len(keys) * val)
    parts = {"test": keys[:n_test], "val": keys[n_test:n_test + n_val], "train": keys[n_test + n_val:]}
    return {s: [groups[k][0] for k in ks] for s, ks in parts.items()}


def _load_background(path: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(path))
    if max(img.size) > 2048:
        img.draft("RGB", (2 * CANVAS, 2 * CANVAS))
    return ImageOps.fit(img.convert("RGB"), (CANVAS, CANVAS), Image.LANCZOS)


def _augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(0.7, 1.3))
    return img


def _scatter_layout(n: int, rng: random.Random, gap: int = 6):
    """Up to n non-overlapping boxes (x, y, w, h) at random sizes and positions."""
    boxes = []
    for _ in range(n):
        for _ in range(200):
            side = rng.randint(130, 300)
            w = h = side
            if rng.random() < 0.3:
                h = int(side * rng.uniform(0.8, 1.25))
            if w > CANVAS or h > CANVAS:
                continue
            x, y = rng.randint(0, CANVAS - w), rng.randint(0, CANVAS - h)
            if all(x + w + gap <= bx or bx + bw + gap <= x or y + h + gap <= by or by + bh + gap <= y
                   for bx, by, bw, bh in boxes):
                boxes.append((x, y, w, h))
                break
    return boxes


def _grid_layout(n: int, rng: random.Random):
    """Tiles of a 2x2 or 3x3 grid, n of them filled; the classic 'concatenated images' layout."""
    k = 2 if n <= 4 else 3
    cell = CANVAS // k
    cells = rng.sample([(r, c) for r in range(k) for c in range(k)], min(n, k * k))
    return [(c * cell, r * cell, cell, cell) for r, c in cells]


def _overlap(a, b) -> int:
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, ix) * max(0, iy)


def _collage_layout(n: int, rng: random.Random, max_overlap: float = 0.12, touch: int = 4):
    """Up to n boxes packed against each other: every box after the first touches or overlaps another,
    covering at most max_overlap of any box's area (objects stay mostly visible)."""
    boxes = []
    for _ in range(n):
        for _ in range(400):
            w = rng.randint(170, 320)
            h = int(w * rng.uniform(0.85, 1.15)) if rng.random() < 0.3 else w
            if h > CANVAS:
                continue
            x, y = rng.randint(0, CANVAS - w), rng.randint(0, CANVAS - h)
            box = (x, y, w, h)
            if any(_overlap(box, b) > max_overlap * min(w * h, b[2] * b[3]) for b in boxes):
                continue
            grown = (x - touch, y - touch, w + 2 * touch, h + 2 * touch)
            if boxes and not any(_overlap(grown, b) for b in boxes):
                continue
            boxes.append(box)
            break
    return boxes


LAYOUTS = {"scatter": _scatter_layout, "grid": _grid_layout, "collage": _collage_layout}
LAYOUT_SHARE = {"scatter": 0.5, "grid": 0.25, "collage": 0.25}


class ObjectSampler:
    """Cycles through a shuffled pool so every source image is used about equally often."""

    def __init__(self, df: pd.DataFrame, rng: random.Random):
        self.rng = rng
        self.by_label = {lbl: list(g.path) for lbl, g in df.groupby("label")}
        self.queues = {lbl: [] for lbl in self.by_label}
        self.label_queue = []

    def _next_label(self):
        if not self.label_queue:
            self.label_queue = list(self.by_label)
            self.rng.shuffle(self.label_queue)
        return self.label_queue.pop()

    def draw(self, n: int) -> list[tuple[str, str]]:
        """n (path, label) pairs with distinct labels."""
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
            picks.append((self.queues[lbl].pop(), lbl))
        return picks


def compose_split(split: str, objects: pd.DataFrame, backgrounds: list[Path], classes: list[str],
                  n_images: int, out: Path, seed: int, min_objs: int = 2, max_objs: int = 6) -> pd.DataFrame:
    """Write out/images/<split>/*.jpg and out/labels/<split>/*.txt; return one row per pasted object."""
    rng = random.Random(f"{seed}-{split}")
    sampler = ObjectSampler(objects, rng)
    cls_idx = {c: i for i, c in enumerate(classes)}
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n_images):
        name = f"{split}_{i:04d}"
        n = rng.randint(min_objs, max_objs)
        layout = rng.choices(list(LAYOUT_SHARE), weights=list(LAYOUT_SHARE.values()))[0]
        boxes = LAYOUTS[layout](n, rng)
        bg = rng.choice(backgrounds)
        canvas = _augment(_load_background(bg), rng)

        lines = []
        for (x, y, w, h), (path, label) in zip(boxes, sampler.draw(len(boxes))):
            tile = _augment(Image.open(path).convert("RGB"), rng).resize((w, h), Image.LANCZOS)
            canvas.paste(tile, (x, y))
            lines.append(f"{cls_idx[label]} {(x + w / 2) / CANVAS:.6f} {(y + h / 2) / CANVAS:.6f} "
                         f"{w / CANVAS:.6f} {h / CANVAS:.6f}")
            rows.append({"image": name, "split": split, "layout": layout,
                         "background": str(bg), "source": path, "label": label,
                         "x": x, "y": y, "w": w, "h": h})
        canvas.save(out / "images" / split / f"{name}.jpg", "JPEG", quality=92)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
    return pd.DataFrame(rows)


def check_no_leakage(manifest: pd.DataFrame) -> dict:
    """Assert no source or background image (by path or by near-duplicate fingerprint) is shared by two splits."""
    report = {}
    for col in ("source", "background"):
        used = manifest[["split", col]].drop_duplicates()
        used = used.assign(fp=used[col].map(fingerprint))
        for key in (col, "fp"):
            per = used.groupby(key).split.nunique()
            shared = per[per > 1]
            report[f"{col}_{'paths' if key == col else 'fingerprints'}_in_2+_splits"] = int(len(shared))
            if len(shared):
                raise AssertionError(f"{len(shared)} {col} images appear in more than one split: {list(shared.index[:5])}")
        report[f"unique_{col}_images"] = used.groupby("split")[col].nunique().to_dict()
    return report


def validate_labels(out: Path, n_classes: int, min_objects: int = 2) -> dict:
    """Check every image/label pair: matching files, 640x640 image, 5 values per line, class index in range,
    box inside the image with positive size, and at least min_objects objects. Raises on the first problem."""
    checked = boxes = 0
    for split in SPLITS:
        images = {f.stem for f in (out / "images" / split).glob("*.jpg")}
        labels = {f.stem for f in (out / "labels" / split).glob("*.txt")}
        if images != labels:
            raise AssertionError(f"{split}: {len(images ^ labels)} images without labels or labels without images")
        for stem in sorted(labels):
            with Image.open(out / "images" / split / f"{stem}.jpg") as im:
                if im.size != (CANVAS, CANVAS):
                    raise AssertionError(f"{split}/{stem}.jpg is {im.size}, expected {CANVAS}x{CANVAS}")
            lines = [l.split() for l in (out / "labels" / split / f"{stem}.txt").read_text().split("\n") if l.strip()]
            if len(lines) < min_objects:
                raise AssertionError(f"{split}/{stem}.txt has {len(lines)} objects")
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
