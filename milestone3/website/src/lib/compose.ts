import { gridShape, TILE, type Grid, type Single } from "./data";

export type Placed = { x1: number; y1: number; x2: number; y2: number; label: string };

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${src}`));
    img.src = src;
  });
}

export async function composeGrid(
  grid: Grid,
  photos: (Single & { label: string })[],
): Promise<{ canvas: HTMLCanvasElement; placed: Placed[] }> {
  const { cols, rows } = gridShape(grid);
  const canvas = document.createElement("canvas");
  canvas.width = cols * TILE;
  canvas.height = rows * TILE;
  const ctx = canvas.getContext("2d")!;
  const imgs = await Promise.all(photos.map((p) => loadImage(p.src)));
  const placed = photos.slice(0, cols * rows).map((p, i) => {
    const x = (i % cols) * TILE;
    const y = Math.floor(i / cols) * TILE;
    ctx.drawImage(imgs[i], x, y, TILE, TILE);
    const [x1, y1, x2, y2] = p.box;
    return { x1: x + x1, y1: y + y1, x2: x + x2, y2: y + y2, label: p.label };
  });
  return { canvas, placed };
}
