export const CANVAS = 640;
export type Layout = "scatter" | "grid" | "collage";
export type Rect = { x: number; y: number; w: number; h: number };

const randint = (a: number, b: number) => a + Math.floor(Math.random() * (b - a + 1));
const uniform = (a: number, b: number) => a + Math.random() * (b - a);

function overlap(a: Rect, b: Rect) {
  const ix = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const iy = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  return Math.max(0, ix) * Math.max(0, iy);
}

function scatter(n: number, gap = 6): Rect[] {
  const boxes: Rect[] = [];
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < 200; t++) {
      const side = randint(130, 300);
      const w = side;
      const h = Math.random() < 0.3 ? Math.round(side * uniform(0.8, 1.25)) : side;
      if (h > CANVAS) continue;
      const r = { x: randint(0, CANVAS - w), y: randint(0, CANVAS - h), w, h };
      const free = boxes.every(
        (b) => r.x + r.w + gap <= b.x || b.x + b.w + gap <= r.x || r.y + r.h + gap <= b.y || b.y + b.h + gap <= r.y,
      );
      if (free) {
        boxes.push(r);
        break;
      }
    }
  }
  return boxes;
}

function grid(n: number): Rect[] {
  const k = n <= 4 ? 2 : 3;
  const cell = Math.floor(CANVAS / k);
  const cells = Array.from({ length: k * k }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, n);
  return cells.map((i) => ({ x: (i % k) * cell, y: Math.floor(i / k) * cell, w: cell, h: cell }));
}

function collage(n: number, maxOverlap = 0.12, touch = 4): Rect[] {
  const boxes: Rect[] = [];
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < 400; t++) {
      const w = randint(170, 320);
      const h = Math.random() < 0.3 ? Math.round(w * uniform(0.85, 1.15)) : w;
      if (h > CANVAS) continue;
      const r = { x: randint(0, CANVAS - w), y: randint(0, CANVAS - h), w, h };
      if (boxes.some((b) => overlap(r, b) > maxOverlap * Math.min(r.w * r.h, b.w * b.h))) continue;
      const grown = { x: r.x - touch, y: r.y - touch, w: r.w + 2 * touch, h: r.h + 2 * touch };
      if (boxes.length && !boxes.some((b) => overlap(grown, b) > 0)) continue;
      boxes.push(r);
      break;
    }
  }
  return boxes;
}

export function layoutBoxes(layout: Layout, n: number): Rect[] {
  return layout === "grid" ? grid(n) : layout === "collage" ? collage(n) : scatter(n);
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${src}`));
    img.src = src;
  });
}

export async function composeScene(
  backgroundSrc: string,
  photos: { src: string; label: string }[],
  layout: Layout,
): Promise<{ canvas: HTMLCanvasElement; placed: (Rect & { label: string })[] }> {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = CANVAS;
  const ctx = canvas.getContext("2d")!;
  const [bg, ...imgs] = await Promise.all([loadImage(backgroundSrc), ...photos.map((p) => loadImage(p.src))]);
  const s = Math.max(CANVAS / bg.width, CANVAS / bg.height);
  ctx.drawImage(bg, (CANVAS - bg.width * s) / 2, (CANVAS - bg.height * s) / 2, bg.width * s, bg.height * s);
  const boxes = layoutBoxes(layout, photos.length);
  const placed = boxes.map((r, i) => {
    ctx.drawImage(imgs[i], r.x, r.y, r.w, r.h);
    return { ...r, label: photos[i].label };
  });
  return { canvas, placed };
}
