"""Evaluate the trained detector on the held-out test split.

Writes results/test_metrics.json (mAP, precision, recall, best-F1 threshold, object/image-level accuracy
overall and per layout), results/per_class_ap.csv, results/test_errors.json, the Ultralytics curves and
confusion matrix, and example detections / failure cases.

    python scripts/03_evaluate_yolo.py                          # models/yolov8s_best.pt
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
DATA = ROOT / "data" / "multi"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8s_best.pt")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold for the accuracy counts")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(args.weights))
    names = model.names
    classes = [names[i] for i in range(len(names))]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=640, batch=16, device=device,
                  project=str(ROOT / "runs"), name="test_eval", exist_ok=True, plots=True)
    box = m.box
    per_class = pd.DataFrame({
        "object_id": [names[i] for i in box.ap_class_index],
        "precision": box.p, "recall": box.r, "AP50": box.ap50, "AP50-95": box.ap,
    }).round(4).sort_values("AP50")
    per_class.to_csv(results_dir / "per_class_ap.csv", index=False)

    layout = pd.read_csv(DATA / "manifest.csv").drop_duplicates("image").set_index("image").layout
    rows, preds, wrong_pairs, missed = [], {}, Counter(), Counter()
    for f in sorted((DATA / "images" / "test").glob("*.jpg")):
        r = model.predict(str(f), conf=args.conf, device=device, agnostic_nms=True, verbose=False)[0]
        pred = [(int(c), float(s), tuple(b)) for c, s, b in
                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        preds[f.stem] = pred
        gt = read_gt(DATA / "labels" / "test" / f"{f.stem}.txt", 640)
        correct, wrong, missed_n, fa, ious = match_image(gt, pred)
        rows.append({"image": f.stem, "layout": layout[f.stem], "n_gt": len(gt), "correct": correct,
                     "wrong_id": wrong, "missed": missed_n, "false_alarm": fa, "ious": ious})
        for c, b in gt:
            hits = [pc for pc, _, pb in pred if iou(pb, b) >= 0.5]
            if not hits:
                missed[names[c]] += 1
            elif c not in hits:
                wrong_pairs[f"{names[c]} -> {names[hits[0]]}"] += 1

    try:  # confidence that maximises the class-averaged F1 curve
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
        f"at_conf_{args.conf}": image_level_summary(rows),
        "by_layout": {k: image_level_summary(g.to_dict("records")) for k, g in df.groupby("layout")},
        "class_tiers": {k: {"count": len(v), "mean_AP50": round(float(per_class.set_index("object_id").AP50[v].mean()), 4)
                            if v else None, "classes": v} for k, v in tiers.items()},
    }
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    df.drop(columns="ious").to_csv(results_dir / "test_per_image.csv", index=False)
    (results_dir / "test_errors.json").write_text(json.dumps(
        {"wrong_id_pairs": dict(wrong_pairs.most_common()), "missed_by_class": dict(missed.most_common())}, indent=2))

    acc = metrics[f"at_conf_{args.conf}"]
    rule = "=" * 70
    print(f"{rule}\nTEST SET EVALUATION ({len(rows)} images, {acc['objects']} objects)\n{rule}")
    print(f"mAP@0.5        {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95   {metrics['mAP50-95']:.4f}")
    print(f"Precision      {metrics['precision']:.4f}")
    print(f"Recall         {metrics['recall']:.4f}")
    if best_f1:
        print(f"Best F1        {best_f1['f1']:.4f} at confidence {best_f1['confidence']}")
    print(f"Inference      {metrics['inference_ms_per_image']} ms/image")
    print(f"{rule}\nAT CONFIDENCE {args.conf}\n{rule}")
    print(f"Correct ID and location   {acc['objects_correct_id_and_location']:>4} / {acc['objects']}  ({acc['object_accuracy']:.1%})")
    print(f"Located, wrong ID         {acc['wrong_id']:>4}")
    print(f"Missed                    {acc['missed']:>4}")
    print(f"False alarms              {acc['false_alarms']:>4}")
    print(f"Fully correct images      {acc['images_fully_correct']:>4} / {acc['images']}  ({acc['image_accuracy']:.1%})")
    print(f"{'layout':<10}{'images':>8}{'objects':>9}{'obj. acc.':>11}{'image acc.':>12}")
    for k, v in metrics["by_layout"].items():
        print(f"{k:<10}{v['images']:>8}{v['objects']:>9}{v['object_accuracy']:>11.1%}{v['image_accuracy']:>12.1%}")
    print(f"{rule}\nLOWEST AP@0.5 CLASSES\n{rule}")
    print(per_class.head(5).to_string(index=False))

    for fname in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
                  "BoxP_curve.png", "BoxR_curve.png"):
        src = ROOT / "runs" / "test_eval" / fname
        if src.exists():
            shutil.copy2(src, results_dir / f"test_{fname}")

    # example detections: ground truth (left) vs prediction (right); the worst images first
    df = df.assign(err=lambda d: d.n_gt - d.correct + d.false_alarm)
    picks = [stem for k in ("scatter", "grid", "collage") for stem in df[df.layout == k].sort_values("image").image[:2]]
    worst = list(df[df.err > 0].sort_values("err", ascending=False).image[:4])
    for tag, ids in (("example_detections", picks), ("failure_cases", worst)):
        if not ids:
            continue
        tiles = []
        for stem in ids:
            img = Image.open(DATA / "images" / "test" / f"{stem}.jpg")
            gt = read_yolo_labels(DATA / "labels" / "test" / f"{stem}.txt", 640, classes)
            pr = [(*b, f"{names[c]} {s:.2f}", c) for c, s, b in preds[stem]]
            tiles += [draw_boxes(img, gt), draw_boxes(img, pr)]
        grid(tiles, cols=4).save(ROOT / "screenshots" / f"{tag}.jpg", quality=90)
    print(f"{rule}\nSaved results/test_metrics.json, per_class_ap.csv, test_errors.json; "
          f"screenshots/example_detections.jpg, failure_cases.jpg")


if __name__ == "__main__":
    main()
