"""Render the console output of the pipeline as terminal-style figures for the report.

Reads runs/make_multi_object.log, runs/train_yolov8s.log (+ results.csv), runs/evaluate.log and
runs/detect.log, and writes screenshots/terminal_{generate,train,evaluate,detect}.png.
"""
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(text: str) -> list[str]:
    lines = []
    for raw in text.replace("\r", "\n").split("\n"):
        line = ANSI.sub("", raw).rstrip()
        if "━" in line or "──" in line:
            continue
        lines.append(line.encode("ascii", "ignore").decode())
    return lines


def render(lines: list[str], out: Path, title: str, max_chars: int = 112):
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    cw, lh = font.getbbox("M")[2], 21
    lines = [l if len(l) <= max_chars else l[:max_chars - 3] + "..." for l in lines]
    width = max(80, max(map(len, lines))) * cw + 48
    height = len(lines) * lh + 76
    img = Image.new("RGB", (width, height), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 34], fill=(48, 48, 48))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([16 + i * 20, 11, 28 + i * 20, 23], fill=c)
    d.text((width // 2, 17), title, font=font, fill=(200, 200, 200), anchor="mm")
    for i, line in enumerate(lines):
        heading = line.startswith("=") or (line.isupper() and not any(ch.isdigit() for ch in line))
        color = (120, 220, 140) if heading else (255, 110, 100) if "WRONG" in line or "missed:" in line else (230, 230, 230)
        d.text((24, 50 + i * lh), line, font=font, fill=color)
    out.parent.mkdir(exist_ok=True)
    img.save(out)
    print(f"-> {out.relative_to(ROOT)}  ({len(lines)} lines)")


def training_lines() -> list[str]:
    log = clean((RUNS / "train_yolov8s.log").read_text(errors="ignore"))
    keep = ("Ultralytics", "Overriding model.yaml", "Model summary", "Transferred", "optimizer: AdamW",
            "Using 1200", "Starting training")
    head = list(dict.fromkeys(l for l in log if l.startswith(keep)))
    with open(RUNS / "yolov8s" / "results.csv") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]
    last = len(rows)
    picked = [r for r in rows if int(float(r["epoch"])) in {1, 2, 3, 5, 10, 15, 20, 30, 40, 50, last}]
    cols = [("epoch", "Epoch", "{:>5.0f}"), ("train/box_loss", "box_loss", "{:>9.3f}"),
            ("train/cls_loss", "cls_loss", "{:>9.3f}"), ("train/dfl_loss", "dfl_loss", "{:>9.3f}"),
            ("metrics/precision(B)", "P", "{:>7.3f}"), ("metrics/recall(B)", "R", "{:>7.3f}"),
            ("metrics/mAP50(B)", "mAP50", "{:>8.3f}"), ("metrics/mAP50-95(B)", "mAP50-95", "{:>9.3f}")]
    table = ["".join(f"{h:>{len(f.format(0))}}" for _, h, f in cols)]
    table += ["".join(f.format(float(r[k])) for k, _, f in cols) for r in picked]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return (["=" * 70, "YOLOV8S TRAINING (transfer learning from COCO)", "=" * 70] + head + [""] + table
            + ["", f"{last} epochs completed; best mAP50-95 {float(best['metrics/mAP50-95(B)']):.4f} at epoch "
               f"{int(float(best['epoch']))} -> models/yolov8s_best.pt"])


def main():
    render(clean((RUNS / "make_multi_object.log").read_text()), ROOT / "screenshots" / "terminal_generate.png",
           "python scripts/01_make_multi_object.py")
    render(training_lines(), ROOT / "screenshots" / "terminal_train.png", "python scripts/02_train_yolo.py")
    ev = clean((RUNS / "evaluate.log").read_text())
    start = next(i for i, l in enumerate(ev) if l.startswith("=" * 10))
    render(ev[start:], ROOT / "screenshots" / "terminal_evaluate.png", "python scripts/03_evaluate_yolo.py")
    render([l for l in clean((RUNS / "detect.log").read_text()) if not l.startswith("Annotated")],
           ROOT / "screenshots" / "terminal_detect.png", "python scripts/04_detect.py data/multi/images/test --limit 10")


if __name__ == "__main__":
    main()
