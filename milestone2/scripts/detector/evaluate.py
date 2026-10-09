from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match_image(gt, pred, thr: float = 0.5):
    """Greedy match by confidence. gt: [(cls, box)], pred: [(cls, conf, box)].

    Returns (correct, wrong_id, missed, false_alarm, ious of correct matches):
    correct = right ID with IoU >= thr, wrong_id = a box found at IoU >= thr but with another ID.
    """
    free = set(range(len(gt)))
    correct = wrong = 0
    ious = []
    for cls, _, box in sorted(pred, key=lambda p: -p[1]):
        best, best_iou = None, thr
        for j in free:
            v = iou(box, gt[j][1])
            if v >= best_iou:
                best, best_iou = j, v
        if best is None:
            continue
        free.discard(best)
        if gt[best][0] == cls:
            correct += 1
            ious.append(best_iou)
        else:
            wrong += 1
    matched = correct + wrong
    return correct, wrong, len(gt) - matched, len(pred) - matched, ious


def read_gt(label_file: Path, width: int, height: int):
    gt = []
    for line in Path(label_file).read_text().split("\n"):
        if line.strip():
            c, xc, yc, w, h = line.split()
            xc, w = float(xc) * width, float(w) * width
            yc, h = float(yc) * height, float(h) * height
            gt.append((int(c), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    return gt


def image_level_summary(rows) -> dict:
    """rows: per-image dicts from match_image plus n_gt."""
    n_gt = sum(r["n_gt"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "images": len(rows),
        "objects": n_gt,
        "objects_correct_id_and_location": correct,
        "object_accuracy": round(correct / n_gt, 4),
        "wrong_id": sum(r["wrong_id"] for r in rows),
        "missed": sum(r["missed"] for r in rows),
        "false_alarms": sum(r["false_alarm"] for r in rows),
        "images_fully_correct": sum(r["correct"] == r["n_gt"] and r["false_alarm"] == 0 for r in rows),
        "image_accuracy": round(np.mean([r["correct"] == r["n_gt"] and r["false_alarm"] == 0 for r in rows]), 4),
        "mean_iou_correct": round(float(np.mean([v for r in rows for v in r["ious"]])), 4),
    }
