"""Export both models to ONNX and the results/sample images for the demo website (../website).

The detector (YOLOv8s) and the classifier (milestone 1 EfficientNet-B0) are exported to ONNX and
checked against PyTorch on test images before anything is written. Then metrics, training history,
class list, sample test images and single-object photos are copied into the site.

    python scripts/05_export_web.py
"""
import argparse
import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from detector.evaluate import iou

ROOT = Path(__file__).resolve().parent.parent
M1 = ROOT.parent / "milestone1"
sys.path.insert(0, str(M1 / "scripts"))

from classifier.data import eval_transform  # noqa: E402
from classifier.predict import load_model  # noqa: E402

DATA = ROOT / "data"


def letterbox(img: Image.Image, size=640):
    w, h = img.size
    s = min(size / w, size / h)
    nw, nh = round(w * s), round(h * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) / 2, (size - nh) / 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (round(px), round(py)))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    return x, s, round(px), round(py)


def onnx_detect(sess, img, conf=0.25, iou_thr=0.7):
    """The same decode + class-agnostic NMS the website runs (website/src/lib/detector.ts)."""
    x, s, px, py = letterbox(img)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    scores = out[4:]
    cls, best = scores.argmax(0), scores.max(0)
    keep = np.where(best >= conf)[0]
    dets = []
    for j in keep[np.argsort(-best[keep])]:
        cx, cy, w, h = out[:4, j]
        box = ((cx - w / 2 - px) / s, (cy - h / 2 - py) / s, (cx + w / 2 - px) / s, (cy + h / 2 - py) / s)
        if all(iou(box, d[2]) < iou_thr for d in dets):
            dets.append((int(cls[j]), float(best[j]), box))
    return dets


def export_detector(weights: Path, out: Path, test_images: list[Path]) -> list[str]:
    model = YOLO(str(weights))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_w = Path(tmp) / "detector.pt"
        shutil.copy2(weights, tmp_w)
        onnx_path = Path(YOLO(str(tmp_w)).export(format="onnx", imgsz=640, opset=17, simplify=True, verbose=False))
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        worst = 1.0
        for f in test_images:
            img = Image.open(f).convert("RGB")
            ref = model.predict(img, conf=0.25, agnostic_nms=True, device="cpu", verbose=False)[0]
            ref = [(int(c), tuple(b)) for c, b in zip(ref.boxes.cls.tolist(), ref.boxes.xyxy.tolist())]
            got = onnx_detect(sess, img)
            if len(ref) != len(got):
                raise SystemExit(f"detector parity failed on {f.name}: {len(ref)} vs {len(got)} boxes")
            for c, b in ref:
                match = max((iou(b, g[2]) for g in got if g[0] == c), default=0)
                worst = min(worst, match)
                if match < 0.98:
                    raise SystemExit(f"detector parity failed on {f.name}: IoU {match:.3f} for class {c}")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx_path, out)
    print(f"detector  -> {out.name} ({out.stat().st_size / 1e6:.1f} MB), "
          f"parity on {len(test_images)} test images: worst box IoU {worst:.4f}")
    return [model.names[i] for i in range(len(model.names))]


def export_classifier(ckpt: Path, out: Path, test_images: list[Path]) -> list[str]:
    model, classes = load_model(ckpt, "cpu")
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "classifier.onnx"
        torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(onnx_path), input_names=["input"],
                          output_names=["logits"], opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tf = eval_transform()
        x = torch.stack([tf(Image.open(f).convert("RGB")) for f in test_images])
        with torch.no_grad():
            ref = model(x).argmax(1).numpy()
        got = np.concatenate([sess.run(None, {"input": x[i:i + 1].numpy()})[0] for i in range(len(x))]).argmax(1)
        if (ref != got).any():
            raise SystemExit(f"classifier parity failed on {(ref != got).sum()} of {len(x)} images")
        shutil.copy2(onnx_path, out)
    print(f"classifier -> {out.name} ({out.stat().st_size / 1e6:.1f} MB), "
          f"parity: same top-1 on {len(test_images)}/{len(test_images)} test images")
    return classes


def save_jpg(src: Path, dst: Path, size=None, quality=85):
    img = Image.open(src).convert("RGB")
    if size:
        img = img.resize((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=quality)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=ROOT / "models" / "yolov8s_best.pt")
    ap.add_argument("--classifier", type=Path, default=M1 / "models" / "EfficientNetB0.pt")
    ap.add_argument("--site", type=Path, default=ROOT.parent / "website")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    public, data_dir = args.site / "public", args.site / "src" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    multi_test = sorted((DATA / "multi" / "images" / "test").glob("*.jpg"))
    singles_test = sorted((DATA / "singles" / "test").glob("*/*.jpg"))
    det_classes = export_detector(args.detector, public / "models" / "detector.onnx", rng.sample(multi_test, 10))
    cls_classes = export_classifier(args.classifier, public / "models" / "classifier.onnx",
                                    rng.sample(singles_test, 50))
    if cls_classes != det_classes:
        print(f"WARNING: classifier has {len(cls_classes)} classes, detector {len(det_classes)}; "
              "retrain milestone 1 on the 73-class split")

    # sample test composites (8 per layout) with their labels
    manifest = pd.read_csv(DATA / "multi" / "manifest.csv")
    test = manifest[manifest.split == "test"].drop_duplicates("image")
    samples = []
    shutil.rmtree(public / "samples", ignore_errors=True)
    for layout in ("scatter", "grid", "collage"):
        for stem in sorted(test[test.layout == layout].image)[:8]:
            save_jpg(DATA / "multi" / "images" / "test" / f"{stem}.jpg", public / "samples" / f"{stem}.jpg")
            labels = (DATA / "multi" / "labels" / "test" / f"{stem}.txt").read_text()
            samples.append({"id": stem, "layout": layout, "src": f"/samples/{stem}.jpg", "labels": labels})

    # single-object test photos (3 per class; the first is the class thumbnail) and test backgrounds
    singles = {}
    shutil.rmtree(public / "singles", ignore_errors=True)
    for d in sorted((DATA / "singles" / "test").iterdir()):
        if d.is_dir():
            files = sorted(d.glob("*.jpg"))
            picks = rng.sample(files, min(3, len(files)))
            singles[d.name] = []
            for f in picks:
                save_jpg(f, public / "singles" / d.name / f.name, quality=88)
                singles[d.name].append(f"/singles/{d.name}/{f.name}")
    shutil.rmtree(public / "backgrounds", ignore_errors=True)
    backgrounds = []
    for src in sorted(map(Path, manifest[manifest.split == "test"].background.unique()))[:24]:
        save_jpg(src, public / "backgrounds" / src.name, quality=85)
        backgrounds.append(f"/backgrounds/{src.name}")

    results = ROOT / "results"
    (public / "figures").mkdir(parents=True, exist_ok=True)
    for f in results.glob("test_*.png"):
        shutil.copy2(f, public / "figures" / f.name)
    metrics = json.loads((results / "test_metrics.json").read_text())
    metrics["per_class"] = pd.read_csv(results / "per_class_ap.csv").to_dict("records")
    metrics["errors"] = json.loads((results / "test_errors.json").read_text())
    stats = json.loads((results / "dataset_stats.json").read_text())
    stats.pop("classes", None)
    metrics["dataset"] = stats
    with open(ROOT / "runs" / "yolov8s" / "results.csv") as fh:
        training = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    m1 = json.loads((M1 / "results" / "model_comparison.json").read_text())
    report = json.loads((M1 / "results" / f"{m1['best_model']}_classification_report.json").read_text())
    m1["per_class_f1"] = {k: round(v["f1-score"], 4) for k, v in report["per_class"].items()}
    m1["num_classes"] = len(cls_classes)

    files = {
        "classes.json": {"detector": det_classes, "classifier": cls_classes},
        "metrics.json": metrics,
        "training.json": training,
        "m1.json": m1,
        "samples.json": {"composites": samples, "singles": singles, "backgrounds": backgrounds},
    }
    for name, obj in files.items():
        (data_dir / name).write_text(json.dumps(obj, indent=1))
    print(f"data       -> {data_dir.relative_to(ROOT.parent)}/ ({', '.join(files)})")
    print(f"images     -> {len(samples)} composites, {sum(map(len, singles.values()))} single photos, "
          f"{len(backgrounds)} backgrounds")


if __name__ == "__main__":
    main()
