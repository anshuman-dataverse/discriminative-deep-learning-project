import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png"}
OBJ_RE = re.compile(r"OBJ(\d{3})", re.IGNORECASE)


def _image_files(folder: Path) -> list[Path]:
    return sorted(f for f in folder.rglob("*")
                  if f.is_file() and f.suffix.lower() in EXTS and not f.name.startswith("."))


def _subdirs(raw_dirs):
    for raw in raw_dirs:
        yield from sorted(p for p in Path(raw).iterdir() if p.is_dir())


def audit(raw_dirs) -> pd.DataFrame:
    rows = []
    for d in _subdirs(raw_dirs):
        files = [f for f in d.rglob("*") if f.is_file()]
        imgs = _image_files(d)
        sizes, modes, bad = Counter(), Counter(), 0
        for f in imgs:
            try:
                with Image.open(f) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except Exception:
                bad += 1
        rows.append({
            "folder": d.name,
            "images": len(imgs),
            "background": sum("bg" in f.stem.lower() for f in imgs),
            "not_224x224": sum(v for k, v in sizes.items() if k != (IMG_SIZE, IMG_SIZE)),
            "sizes": ", ".join(f"{w}x{h}" for (w, h), _ in sizes.most_common(2)),
            "non_RGB": sum(v for k, v in modes.items() if k != "RGB"),
            "unreadable": bad,
            "stray_files": len(files) - len(imgs),
            "name_ok": bool(re.fullmatch(r"images_OBJ\d{3}", d.name)),
        })
    return pd.DataFrame(rows)


def normalize(raw_dirs, out: Path) -> pd.DataFrame:
    """Rebuild `out` as images_OBJ###/ folders of 224x224 RGB OBJ###_###.jpg / OBJ###_bg_###.jpg."""
    folders = defaultdict(list)
    for d in _subdirs(raw_dirs):
        m = OBJ_RE.search(d.name)
        if m:
            folders[f"OBJ{m.group(1)}"].append(d)
        else:
            print(f"skipped folder without an OBJ id: {d}")

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    summary = []
    for obj, dirs in sorted(folders.items()):
        dest = out / f"images_{obj}"
        dest.mkdir(parents=True)
        n_obj = n_bg = 0
        for f in (f for d in dirs for f in _image_files(d)):
            try:
                img = ImageOps.exif_transpose(Image.open(f))
                if max(img.size) > 2048:
                    img.draft("RGB", (2 * IMG_SIZE, 2 * IMG_SIZE))
                img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            except Exception as e:
                print(f"unreadable, skipped: {f} ({e})")
                continue
            if "bg" in f.stem.lower():
                n_bg += 1
                name = f"{obj}_bg_{n_bg:03d}.jpg"
            else:
                n_obj += 1
                name = f"{obj}_{n_obj:03d}.jpg"
            img.save(dest / name, "JPEG", quality=95)
        summary.append({"object": obj, "object_imgs": n_obj, "background_imgs": n_bg})
    return pd.DataFrame(summary)


def remove_duplicates(data_dir: Path) -> tuple[list[str], int]:
    """Delete near-identical photos (16x16 grayscale fingerprint), keeping the object copy over a background one.

    Returns the removed paths and the number of duplicate groups spanning two different objects.
    """
    groups = defaultdict(list)
    for f in sorted(Path(data_dir).rglob("*.jpg")):
        with Image.open(f) as im:
            groups[hashlib.md5(im.convert("L").resize((16, 16)).tobytes()).hexdigest()].append(f)

    removed, cross_class = [], 0
    for files in groups.values():
        if len(files) < 2:
            continue
        cross_class += len({f.parent.name for f in files}) > 1
        files.sort(key=lambda p: ("_bg_" in p.name, p.name))
        for f in files[1:]:
            removed.append(str(f))
            f.unlink()
    return removed, cross_class


def make_split(data_dir: Path, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Stratified per-class train/val/test split of the object images (background images excluded)."""
    rows = [{"path": str(f), "label": d.name.removeprefix("images_")}
            for d in sorted(Path(data_dir).iterdir()) if d.is_dir()
            for f in sorted(d.glob("*.jpg")) if "_bg_" not in f.name]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no images found under {data_dir}")

    rng = random.Random(seed)
    df["split"] = ""
    for idx in df.groupby("label").groups.values():
        idx = list(idx)
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test))
        n_val = max(1, round(len(idx) * val))
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
        df.loc[idx[n_test + n_val:], "split"] = "train"
    return df


SPLITS = ("train", "val", "test")


def write_split(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Copy each image to data_dir/<split>/<label>/ and return the split with the new paths."""
    data_dir = Path(data_dir)
    for split in SPLITS:
        shutil.rmtree(data_dir / split, ignore_errors=True)
    paths = []
    for r in df.itertuples():
        dest = data_dir / r.split / r.label / Path(r.path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.path, dest)
        paths.append(str(dest))
    return df.assign(path=paths)


def load_split(data_dir: Path) -> pd.DataFrame:
    """Read the split back from data_dir/<split>/<label>/*.jpg."""
    rows = [{"path": str(f), "label": d.name, "split": split}
            for split in SPLITS
            for d in sorted((Path(data_dir) / split).iterdir()) if d.is_dir()
            for f in sorted(d.glob("*.jpg"))]
    if not rows:
        raise SystemExit(f"no split under {data_dir}/; run scripts/02_split_data.py first")
    return pd.DataFrame(rows)


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor):
    return (t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)


class ImageDataset(Dataset):
    """Holds every image in memory; the dataset is a few thousand 224x224 images."""

    def __init__(self, df: pd.DataFrame, class_to_idx: dict, transform):
        self.images = [Image.open(p).convert("RGB") for p in df.path]
        self.targets = [class_to_idx[label] for label in df.label]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def build_loaders(split: pd.DataFrame, batch_size: int = 32, workers: int = 0):
    classes = sorted(split.label.unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    loaders = {
        name: DataLoader(
            ImageDataset(split[split.split == name], class_to_idx,
                         train_transform() if name == "train" else eval_transform()),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        for name in ("train", "val", "test")
    }
    return loaders, classes


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
