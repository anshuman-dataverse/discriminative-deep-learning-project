"""Load a trained checkpoint and identify objects in images."""
from pathlib import Path

import torch
from PIL import Image

from .data import eval_transform
from .models import build_model


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"]


def _as_rgb(image) -> Image.Image:
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    return img.convert("RGB")


@torch.no_grad()
def predict_proba(model, images, device, batch_size: int = 64) -> torch.Tensor:
    """Class probabilities for a list of paths or PIL images, shape (N, num_classes)."""
    tf = eval_transform()
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([tf(_as_rgb(im)) for im in images[i:i + batch_size]]).to(device)
        out.append(model(x).softmax(1).cpu())
    return torch.cat(out)


def identify(model, classes, image, device, k: int = 3) -> list[tuple[str, float]]:
    """Top-k (Object ID, probability) for a single path or PIL image."""
    probs = predict_proba(model, [image], device)[0]
    top = probs.topk(min(k, len(classes)))
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]

