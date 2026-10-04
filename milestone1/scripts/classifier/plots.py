from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_curves(hist: pd.DataFrame, path: Path, title: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in ((a, "loss"), (b, "acc")):
        ax.plot(hist.epoch, hist[f"train_{metric}"], label="train")
        ax.plot(hist.epoch, hist[f"val_{metric}"], label="val")
        ax.set_title("Loss" if metric == "loss" else "Accuracy")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list, path: Path | None, title: str):
    n = len(classes)
    size = max(6, n * 0.28)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
    fs = 8 if n <= 30 else 6
    ax.set_xticks(range(n), classes, rotation=90, fontsize=fs)
    ax.set_yticks(range(n), classes, fontsize=fs)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_comparison(metrics: pd.DataFrame, path: Path | None = None):
    best = metrics.model.iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
    a.bar(metrics.model, metrics.test_acc, color=["#55A868" if m == best else "#4C72B0" for m in metrics.model])
    a.set_ylim(0, 1.05)
    a.set_title("Test accuracy")
    for i, v in enumerate(metrics.test_acc):
        a.text(i, v + 0.01, f"{v:.1%}", ha="center")
    b.scatter(metrics.inference_ms_per_img, metrics.test_acc, s=metrics.params_M * 25 + 30)
    for r in metrics.itertuples():
        b.annotate(r.model, (r.inference_ms_per_img, r.test_acc), xytext=(6, 4), textcoords="offset points")
    b.set_xlabel("inference time (ms / image)")
    b.set_ylabel("test accuracy")
    b.set_title("Accuracy vs speed (bubble size = parameters)")
    b.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
