"""Object boxes in single-object photos from two open-vocabulary detectors (Grounding DINO and OWLv2)."""
import torch
from PIL import Image
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection,
                          Owlv2Processor)

GDINO = "IDEA-Research/grounding-dino-tiny"
OWL = "google/owlv2-base-patch16-ensemble"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class GroundingDino:
    def __init__(self, dev: str):
        self.dev = dev
        self.proc = AutoProcessor.from_pretrained(GDINO)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(dev).eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], prompts: list[str]):
        """Highest-scoring (score, (x1, y1, x2, y2)) per image, or None."""
        inputs = self.proc(images=images, text=[f"{p}." for p in prompts], return_tensors="pt", padding=True).to(self.dev)
        out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.2, text_threshold=0.2,
            target_sizes=[im.size[::-1] for im in images])
        return [_best(r["scores"], r["boxes"]) for r in res]


class Owl:
    def __init__(self, dev: str):
        self.dev = dev
        self.proc = Owlv2Processor.from_pretrained(OWL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL).to(dev).eval()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], prompts: list[str]):
        inputs = self.proc(images=images, text=[[f"a photo of a {p}"] for p in prompts], return_tensors="pt").to(self.dev)
        out = self.model(**inputs)
        side = [max(im.size) for im in images]
        res = self.proc.post_process_grounded_object_detection(out, threshold=0.1, target_sizes=[(s, s) for s in side])
        return [_best(r["scores"], r["boxes"]) for r in res]


def _best(scores, boxes):
    if len(scores) == 0:
        return None
    i = int(scores.argmax())
    return float(scores[i]), tuple(round(float(v), 1) for v in boxes[i])
