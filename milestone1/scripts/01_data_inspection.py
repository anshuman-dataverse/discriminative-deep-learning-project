"""Audit the raw shared-Drive download: image counts, sizes, color modes, naming and stray files.

    python scripts/01_data_inspection.py                 # reads data/raw/*/
"""
import argparse
from pathlib import Path

import pandas as pd

from classifier.data import audit

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="*", help="extracted Drive download folders (default: data/raw/*)")
    raw = ap.parse_args().raw or sorted(p for p in (ROOT / "data" / "raw").iterdir() if p.is_dir())

    report = audit(raw)
    issues = report[(report.images != 100) | (report.background != 5) | (report.not_224x224 > 0)
                    | (report.non_RGB > 0) | (report.unreadable > 0) | (report.stray_files > 0) | ~report.name_ok]
    pd.set_option("display.width", 200)
    print(f"{len(report)} folders, {report.images.sum()} images, {report.background.sum()} background\n")
    print("Folders with issues:")
    print(issues.to_string(index=False))


if __name__ == "__main__":
    main()
