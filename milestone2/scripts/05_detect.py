"""Detect every object in multi-object image(s): prints Object ID, confidence and box, saves an annotated copy.

If an image has a YOLO label file (images/<split>/x.jpg -> labels/<split>/x.txt), each detection is also
marked correct or wrong, and missed objects are listed. A summary closes the run. The default confidence
threshold is the one 04_evaluate_yolo.py chose on the validation split (0.5 if it has not run yet).
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from detector.evaluate import iou, read_gt
from detector.plots import draw_boxes

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8s_best.pt")
    metrics = ROOT / "results" / "test_metrics.json"
    ap.add_argument("--conf", type=float, default=json.loads(metrics.read_text()).get("conf", 0.5) if metrics.exists() else 0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "detect")
    ap.add_argument("--imgsz", type=int, default=1120)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    args = ap.parse_args()

    files = [f for p in args.images for f in (sorted(p.iterdir()) if p.is_dir() else [p])
             if f.suffix.lower() in EXTS][:args.limit]
    device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)

    confs, n_correct, n_labelled, n_missed = [], 0, 0, 0
    for f in files:
        r = model.predict(str(f), conf=args.conf, imgsz=args.imgsz, device=device, agnostic_nms=True,
                          verbose=False)[0]
        dets = sorted(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()),
                      key=lambda d: (d[2][1], d[2][0]))
        label = f.parent.parent.parent / "labels" / f.parent.name / f"{f.stem}.txt"
        gt = read_gt(label, r.orig_shape[1], r.orig_shape[0]) if label.exists() else None
        print(f"\n{f.name}: {len(dets)} object(s)")
        print(f"  {'Object ID':<10} {'conf':>5}   x1    y1    x2    y2" + ("   status" if gt is not None else ""))
        for c, s, (x1, y1, x2, y2) in dets:
            line = f"  {model.names[int(c)]:<10} {s:5.2f} {x1:5.0f} {y1:5.0f} {x2:5.0f} {y2:5.0f}"
            if gt is not None:
                ok = any(gc == int(c) and iou((x1, y1, x2, y2), gb) >= 0.5 for gc, gb in gt)
                n_correct += ok
                line += "   correct" if ok else "   WRONG"
            confs.append(s)
            print(line)
        if gt is not None:
            n_labelled += len(gt)
            miss = [model.names[gc] for gc, gb in gt
                    if not any(int(c) == gc and iou(b, gb) >= 0.5 for c, _, b in dets)]
            n_missed += len(miss)
            if miss:
                print(f"  missed: {', '.join(miss)}")
        boxes = [(*b, f"{model.names[int(c)]} {s:.2f}", int(c)) for c, s, b in dets]
        draw_boxes(Image.open(f), boxes).save(args.out / f"{f.stem}_detected.jpg", quality=92)
    rule = "=" * 70
    print(f"\n{rule}\nDETECTION SUMMARY\n{rule}")
    print(f"Images processed          {len(files)}")
    print(f"Objects detected          {len(confs)}")
    print(f"Average objects / image   {len(confs) / max(len(files), 1):.1f}")
    if confs:
        print(f"Average confidence        {sum(confs) / len(confs):.2%}")
    if n_labelled:
        print(f"Correct detections        {n_correct} / {len(confs)}")
        print(f"Missed objects            {n_missed} / {n_labelled}")
    print(f"Annotated images saved to {args.out}/")


if __name__ == "__main__":
    main()
