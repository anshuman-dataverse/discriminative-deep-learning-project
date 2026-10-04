"""Compare the trained models and pick the best one.

    python scripts/04_evaluate_models.py

Writes results/model_comparison.{json,png}.
"""
from pathlib import Path

from classifier.compare import load_metrics, save_comparison
from classifier.plots import plot_comparison

ROOT = Path(__file__).resolve().parent.parent


def main():
    results = ROOT / "results"
    df = load_metrics(results)
    save_comparison(df, results)
    plot_comparison(df, results / "model_comparison.png")
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nBest model: {best.model} | test accuracy {best.test_acc:.2%} | macro F1 {best.test_macro_f1:.3f}")


if __name__ == "__main__":
    main()
