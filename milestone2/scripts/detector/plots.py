import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size=16):
    for name in ("Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_boxes(img: Image.Image, boxes, width: int = 3) -> Image.Image:
    """boxes: iterable of (x1, y1, x2, y2, text, class_index)."""
    img = img.convert("RGB").copy()
    d, font = ImageDraw.Draw(img), _font()
    for x1, y1, x2, y2, text, ci in boxes:
        c = color(ci)
        d.rectangle([x1, y1, x2, y2], outline=c, width=width)
        tw, th = d.textbbox((0, 0), text, font=font)[2:]
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1
        d.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=c)
        d.text((x1 + 3, ty + 1), text, fill=(0, 0, 0), font=font)
    return img


def caption(img: Image.Image, text: str):
    d, font = ImageDraw.Draw(img), _font(15)
    tw, th = d.textbbox((0, 0), text, font=font)[2:]
    d.rectangle([0, img.height - th - 8, tw + 8, img.height], fill="white")
    d.text((4, img.height - th - 5), text, fill="black", font=font)


def read_yolo_labels(label_file: Path, width: int, height: int, classes: list[str]):
    boxes = []
    for line in Path(label_file).read_text().split("\n"):
        if line.strip():
            ci, xc, yc, w, h = line.split()
            ci, xc, w, yc, h = int(ci), float(xc) * width, float(w) * width, float(yc) * height, float(h) * height
            boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, classes[ci], ci))
    return boxes


def grid(images: list[Image.Image], cols: int, cell: int = 480) -> Image.Image:
    rows = -(-len(images) // cols)
    out = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, im in enumerate(images):
        im = im.copy()
        im.thumbnail((cell - 8, cell - 8))
        x, y = (i % cols) * cell + (cell - im.width) // 2, (i // cols) * cell + (cell - im.height) // 2
        out.paste(im, (x, y))
    return out
