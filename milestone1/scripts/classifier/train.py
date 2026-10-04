"""Training and test-set evaluation for the single-object ID models.

For each model writes models/<Model>.pt and results/<Model>_{classification_report.json, training_history.png, confusion_matrix.png}.
"""
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from .models import DEFAULTS, build_model, count_params
from .plots import plot_confusion, plot_curves


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        targets.append(y)
    return torch.cat(logits), torch.cat(targets)


@torch.no_grad()
def time_inference(model, device, reps: int = 50) -> float:
    """Milliseconds per image at batch size 1."""
    model.eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(device.type, lambda: None)
    for _ in range(10):
        model(x)
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        model(x)
    sync()
    return (time.perf_counter() - t) / reps * 1000


def train_model(name, loaders, classes, device, models_dir: Path, results_dir: Path,
                epochs=None, lr=None, patience=6, seed=42) -> dict:
    """Train with early stopping on val accuracy, then score the best checkpoint once on the test set."""
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"{name}.pt"
    epochs = epochs or DEFAULTS[name]["epochs"]
    lr = lr or DEFAULTS[name]["lr"]

    torch.manual_seed(seed)
    model = build_model(name, len(classes)).to(device)
    n_params = count_params(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"=== {name}: {n_params / 1e6:.2f}M params, {epochs} epochs, lr={lr}, device={device}")
    hist, best_acc, bad_epochs, t0 = [], -1.0, 0, time.time()
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, loaders["train"], device, criterion, optimizer)
        vl, va = run_epoch(model, loaders["val"], device, criterion)
        scheduler.step()
        hist.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
        improved = va > best_acc
        if improved:
            best_acc, bad_epochs = va, 0
            torch.save({"model": name, "classes": classes, "state_dict": model.state_dict()}, ckpt)
        else:
            bad_epochs += 1
        print(f"ep {ep:3d}  train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}{' *' if improved else ''}")
        if bad_epochs >= patience:
            print(f"early stop after {patience} epochs without improvement")
            break
    train_time = time.time() - t0
    hist = pd.DataFrame(hist)
    plot_curves(hist, results_dir / f"{name}_training_history.png", name)

    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    logits, y = predict_loader(model, loaders["test"], device)
    pred = logits.argmax(1)
    labels = list(range(len(classes)))
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
    plot_confusion(confusion_matrix(y, pred, labels=labels), classes,
                   results_dir / f"{name}_confusion_matrix.png", f"{name}: test confusion matrix (row-normalized)")

    k = min(5, len(classes))
    metrics = {
        "model": name,
        "params_M": round(n_params / 1e6, 2),
        "epochs_run": len(hist),
        "best_val_acc": round(best_acc, 4),
        "test_acc": round((pred == y).float().mean().item(), 4),
        "test_top5": round((logits.topk(k, 1).indices == y[:, None]).any(1).float().mean().item(), 4),
        "test_macro_precision": round(p, 4),
        "test_macro_recall": round(r, 4),
        "test_macro_f1": round(f1, 4),
        "train_time_min": round(train_time / 60, 2),
        "inference_ms_per_img": round(time_inference(model, device), 2),
        "num_classes": len(classes),
        "n_train": len(loaders["train"].dataset),
        "n_val": len(loaders["val"].dataset),
        "n_test": len(loaders["test"].dataset),
        "lr": lr,
        "device": str(device),
    }
    out = {"summary": metrics,
           "per_class": {c: report[c] for c in classes},
           "history": hist.round(4).to_dict("records")}
    (results_dir / f"{name}_classification_report.json").write_text(json.dumps(out, indent=2))
    print(f"-> test accuracy {metrics['test_acc']:.2%}, macro F1 {metrics['test_macro_f1']:.3f}\n")
    return metrics

