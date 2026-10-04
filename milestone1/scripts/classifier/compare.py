import json
from pathlib import Path

import pandas as pd

COLUMNS = ["model", "params_M", "epochs_run", "best_val_acc", "test_acc", "test_top5", "test_macro_precision",
           "test_macro_recall", "test_macro_f1", "train_time_min", "inference_ms_per_img"]


def load_report(results: Path, model: str) -> dict:
    return json.loads((Path(results) / f"{model}_classification_report.json").read_text())


def load_metrics(results: Path) -> pd.DataFrame:
    """One row per trained model, best first (macro F1, then accuracy)."""
    rows = [json.loads(p.read_text())["summary"] for p in sorted(Path(results).glob("*_classification_report.json"))]
    if not rows:
        raise SystemExit(f"no *_classification_report.json under {results}/; train first")
    return (pd.DataFrame(rows)[COLUMNS]
            .sort_values(["test_macro_f1", "test_acc"], ascending=False)
            .reset_index(drop=True))


def save_comparison(df: pd.DataFrame, results: Path):
    out = {"best_model": df.model.iloc[0], "models": df.to_dict("records")}
    (Path(results) / "model_comparison.json").write_text(json.dumps(out, indent=2))
