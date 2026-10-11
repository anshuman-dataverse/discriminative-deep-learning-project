"""Evaluate the trained detector on the held-out test split.

The confidence threshold for the object/image-level counts is the one with the best object-level F1 on the
validation split; it is then applied unchanged to the test split.
Writes results/test_metrics.json (mAP, precision, recall, best-F1 threshold, chosen threshold, object/image-level
accuracy overall and per grid size), results/per_class_ap.csv, results/test_errors.json, the Ultralytics curves
and confusion matrix, and example detections / failure cases.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from detector.evaluate import image_level_summary, iou, match_image, read_gt
from detector.plots import draw_boxes, grid, read_yolo_labels

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [round(0.25 + 0.05 * i, 2) for i in range(11)]


def predict_split(model, data: Path, split: str, imgsz: int, device: str):
    """{image stem: (ground truth, predictions at confidence >= min(CANDIDATES))} for one split."""
    out = {}
    for f in sorted((data / "images" / split).glob("*.jpg")):
        r = model.predict(str(f), conf=min(CANDIDATES), imgsz=imgsz, device=device, agnostic_nms=True,
                          verbose=False)[0]
        pred = [(int(c), float(s), tuple(b)) for c, s, b in
                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        out[f.stem] = (read_gt(data / "labels" / split / f"{f.stem}.txt", r.orig_shape[1], r.orig_shape[0]), pred)
    return out


def choose_conf(val: dict) -> tuple[float, dict]:
    scores = {}
    for t in CANDIDATES:
        rows = []
        for gt, pred in val.values():
            c, w, m, fa, _ = match_image(gt, [p for p in pred if p[1] >= t])
            rows.append({"n_gt": len(gt), "correct": c, "wrong_id": w, "missed": m, "false_alarm": fa, "ious": []})
        s = image_level_summary(rows)
        tp = s["objects_correct_id_and_location"]
        f1 = 2 * tp / (2 * tp + 2 * s["wrong_id"] + s["missed"] + s["false_alarms"])
        scores[t] = {"f1": round(f1, 4), "image_accuracy": s["image_accuracy"], "object_accuracy": s["object_accuracy"]}
    best = max(CANDIDATES, key=lambda t: scores[t]["f1"])
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8s_best.pt")
    ap.add_argument("--conf", type=float, default=None, help="fixed threshold instead of choosing it on validation")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "multi")
    ap.add_argument("--imgsz", type=int, default=1120)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    DATA = args.data
    model = YOLO(str(args.weights))
    names = model.names
    classes = [names[i] for i in range(len(names))]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=args.imgsz, batch=args.batch, device=device,
                  project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    box = m.box
    per_class = pd.DataFrame({
        "object_id": [names[i] for i in box.ap_class_index],
        "precision": box.p, "recall": box.r, "AP50": box.ap50, "AP50-95": box.ap,
    }).round(4).sort_values("AP50")
    per_class.to_csv(results_dir / "per_class_ap.csv", index=False)

    if args.conf is None:
        conf, val_scores = choose_conf(predict_split(model, DATA, "val", args.imgsz, device))
    else:
        conf, val_scores = args.conf, None
    grid_of = pd.read_csv(DATA / "manifest.csv").drop_duplicates("image").set_index("image").grid
    rows, preds, wrong_pairs, missed = [], {}, Counter(), Counter()
    for stem, (gt, pred) in predict_split(model, DATA, "test", args.imgsz, device).items():
        pred = [p for p in pred if p[1] >= conf]
        preds[stem] = pred
        correct, wrong, missed_n, fa, ious = match_image(gt, pred)
        rows.append({"image": stem, "grid": grid_of[stem], "n_gt": len(gt), "correct": correct,
                     "wrong_id": wrong, "missed": missed_n, "false_alarm": fa, "ious": ious})
        for c, b in gt:
            hits = [pc for pc, _, pb in pred if iou(pb, b) >= 0.5]
            if not hits:
                missed[names[c]] += 1
            elif c not in hits:
                wrong_pairs[f"{names[c]} -> {names[hits[0]]}"] += 1

    try:
        f1 = box.f1_curve.mean(0)
        best_f1 = {"confidence": round(float(box.px[f1.argmax()]), 3), "f1": round(float(f1.max()), 4)}
    except AttributeError:
        best_f1 = None
    tiers = {name: per_class[cond].object_id.tolist() for name, cond in (
        ("excellent (AP50 >= 0.95)", per_class.AP50 >= 0.95),
        ("good (0.80-0.95)", (per_class.AP50 >= 0.80) & (per_class.AP50 < 0.95)),
        ("needs improvement (< 0.80)", per_class.AP50 < 0.80))}
    df = pd.DataFrame(rows)

    metrics = {
        "weights": args.weights.name,
        "test_images": len(rows),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "inference_ms_per_image": round(m.speed["inference"], 2),
        "best_f1": best_f1,
        "conf": conf,
        "conf_selection": {"split": "val", "candidates": val_scores} if val_scores else {"split": "fixed"},
        "at_conf": image_level_summary(rows),
        "by_grid": {k: image_level_summary(g.to_dict("records")) for k, g in df.groupby("grid")},
        "class_tiers": {k: {"count": len(v), "mean_AP50": round(float(per_class.set_index("object_id").AP50[v].mean()), 4)
                            if v else None, "classes": v} for k, v in tiers.items()},
    }
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    df.drop(columns="ious").to_csv(results_dir / "test_per_image.csv", index=False)
    (results_dir / "test_errors.json").write_text(json.dumps(
        {"wrong_id_pairs": dict(wrong_pairs.most_common()), "missed_by_class": dict(missed.most_common())}, indent=2))

    acc = metrics["at_conf"]
    rule = "=" * 70
    print(f"{rule}\nTEST SET EVALUATION ({len(rows)} images, {acc['objects']} objects)\n{rule}")
    print(f"mAP@0.5        {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   {metrics['mAP50-95']:.4f}")
    print(f"Precision      {metrics['precision']:.4f}")
    print(f"Recall         {metrics['recall']:.4f}")
    if best_f1:
        print(f"Best F1        {best_f1['f1']:.4f} at confidence {best_f1['confidence']}")
    print(f"Inference      {metrics['inference_ms_per_image']} ms/image")
    if val_scores:
        print(f"{rule}\nCONFIDENCE THRESHOLD CHOSEN ON VALIDATION\n{rule}")
        print(f"{'conf':>6}{'val F1':>9}{'val obj. acc.':>15}{'val image acc.':>16}")
        for t, v in val_scores.items():
            print(f"{t:>6.2f}{v['f1']:>9.4f}{v['object_accuracy']:>15.1%}{v['image_accuracy']:>16.1%}"
                  + ("   <- chosen" if t == conf else ""))
    print(f"{rule}\nTEST SET AT CONFIDENCE {conf}\n{rule}")
    print(f"Correct ID and location   {acc['objects_correct_id_and_location']:>4} / {acc['objects']}  ({acc['object_accuracy']:.1%})")
    print(f"Located, wrong ID         {acc['wrong_id']:>4}")
    print(f"Missed                    {acc['missed']:>4}")
    print(f"False alarms              {acc['false_alarms']:>4}")
    print(f"Fully correct images      {acc['images_fully_correct']:>4} / {acc['images']}  ({acc['image_accuracy']:.1%})")
    print(f"{'grid':<10}{'images':>8}{'objects':>9}{'obj. acc.':>11}{'image acc.':>12}")
    for k, v in metrics["by_grid"].items():
        print(f"{k:<10}{v['images']:>8}{v['objects']:>9}{v['object_accuracy']:>11.1%}{v['image_accuracy']:>12.1%}")
    print(f"{rule}\nLOWEST AP@0.5 CLASSES\n{rule}")
    print(per_class.head(5).to_string(index=False))

    for fname in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
                  "BoxP_curve.png", "BoxR_curve.png"):
        src = ROOT / "runs" / "test_eval" / fname
        if src.exists():
            shutil.copy2(src, results_dir / f"test_{fname}")

    df = df.assign(err=lambda d: d.n_gt - d.correct + d.false_alarm)
    picks = [stem for k in sorted(df.grid.unique()) for stem in df[df.grid == k].sort_values("image").image[:1]]
    worst = list(df[df.err > 0].sort_values("err", ascending=False).image[:4])
    for tag, ids in (("example_detections", picks), ("failure_cases", worst)):
        if not ids:
            continue
        tiles = []
        for stem in ids:
            img = Image.open(DATA / "images" / "test" / f"{stem}.jpg")
            gt = read_yolo_labels(DATA / "labels" / "test" / f"{stem}.txt", *img.size, classes)
            pr = [(*b, f"{names[c]} {s:.2f}", c) for c, s, b in preds[stem]]
            tiles += [draw_boxes(img, gt), draw_boxes(img, pr)]
        grid(tiles, cols=2, cell=720).save(ROOT / "screenshots" / f"{tag}.jpg", quality=90)
    print(f"{rule}\nSaved results/test_metrics.json, per_class_ap.csv, test_errors.json; "
          f"screenshots/example_detections.jpg, failure_cases.jpg")


if __name__ == "__main__":
    main()
