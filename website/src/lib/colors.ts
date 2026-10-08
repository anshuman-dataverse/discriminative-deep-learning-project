// Same per-class box colours as milestone2/scripts/detector/plots.py (golden-ratio hue steps).
export function classColor(i: number): string {
  const h = ((i * 0.618034) % 1) * 360;
  return `hsl(${h.toFixed(0)} 85% 52%)`;
}
