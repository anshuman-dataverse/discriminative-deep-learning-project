export type Label = { c: number; x1: number; y1: number; x2: number; y2: number };

export function parseLabels(text: string): Label[] {
  return text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((l) => {
      const [c, xc, yc, w, h] = l.split(/\s+/).map(Number);
      return { c, x1: xc - w / 2, y1: yc - h / 2, x2: xc + w / 2, y2: yc + h / 2 };
    });
}
