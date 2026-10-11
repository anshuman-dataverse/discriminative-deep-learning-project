import type { InferenceSession } from "onnxruntime-web";
import { createSession, loadOrt, type Backend } from "./ort";

export const DETECTOR_SIZE = 1120;
const IOU_NMS = 0.7;
const MAX_DET = 100;
const SHOW_CONFIDENCE = 0.5;
const SMALL_FILL = 0.6;

export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Detection = Box & { classIndex: number; label: string; confidence: number };
export type DetectResult = { detections: Detection[]; ms: number; backend: Backend; rescaled: boolean };

let loaded: Promise<{ session: InferenceSession; backend: Backend }> | null = null;

export function loadDetector(onProgress?: (f: number) => void) {
  loaded ??= createSession("/models/detector.onnx", [1, 3, DETECTOR_SIZE, DETECTOR_SIZE], onProgress).catch((e) => {
    loaded = null;
    throw e;
  });
  return loaded;
}

type Letterbox = { data: Float32Array; scale: number; padX: number; padY: number };

function letterbox(source: CanvasImageSource, width: number, height: number, fill = 1): Letterbox {
  const size = DETECTOR_SIZE;
  const scale = Math.min(size / width, size / height) * fill;
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);
  const padX = Math.round((size - w) / 2);
  const padY = Math.round((size - h) / 2);
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(source, padX, padY, w, h);
  const { data: rgba } = ctx.getImageData(0, 0, size, size);
  const plane = size * size;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = rgba[i * 4] / 255;
    data[plane + i] = rgba[i * 4 + 1] / 255;
    data[2 * plane + i] = rgba[i * 4 + 2] / 255;
  }
  return { data, scale, padX, padY };
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

async function runPass(
  source: CanvasImageSource,
  width: number,
  height: number,
  classes: string[],
  minConfidence: number,
  fill: number,
): Promise<{ detections: Detection[]; ms: number; backend: Backend }> {
  const [{ session, backend }, ort] = await Promise.all([loadDetector(), loadOrt()]);
  const lb = letterbox(source, width, height, fill);
  const t0 = performance.now();
  const input = new ort.Tensor("float32", lb.data, [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]);
  const out = (await session.run({ [session.inputNames[0]]: input }))[session.outputNames[0]];
  const ms = performance.now() - t0;

  const [, channels, n] = out.dims as number[];
  const nc = channels - 4;
  const v = out.data as Float32Array;
  const candidates: Detection[] = [];
  for (let j = 0; j < n; j++) {
    let best = 0;
    let bestC = 0;
    for (let c = 0; c < nc; c++) {
      const s = v[(4 + c) * n + j];
      if (s > best) {
        best = s;
        bestC = c;
      }
    }
    if (best < minConfidence) continue;
    const cx = v[j], cy = v[n + j], w = v[2 * n + j], h = v[3 * n + j];
    const unmap = (x: number, pad: number, max: number) => Math.min(max, Math.max(0, (x - pad) / lb.scale));
    candidates.push({
      x1: unmap(cx - w / 2, lb.padX, width),
      y1: unmap(cy - h / 2, lb.padY, height),
      x2: unmap(cx + w / 2, lb.padX, width),
      y2: unmap(cy + h / 2, lb.padY, height),
      classIndex: bestC,
      label: classes[bestC] ?? `class ${bestC}`,
      confidence: best,
    });
  }

  candidates.sort((a, b) => b.confidence - a.confidence);
  const kept: Detection[] = [];
  for (const d of candidates) {
    if (kept.every((k) => iou(k, d) < IOU_NMS)) kept.push(d);
    if (kept.length >= MAX_DET) break;
  }
  return { detections: kept, ms, backend };
}

export async function detect(
  source: CanvasImageSource,
  width: number,
  height: number,
  classes: string[],
  minConfidence = 0.25,
): Promise<DetectResult> {
  const full = await runPass(source, width, height, classes, minConfidence, 1);
  if (full.detections.some((d) => d.confidence >= SHOW_CONFIDENCE)) return { ...full, rescaled: false };
  const small = await runPass(source, width, height, classes, minConfidence, SMALL_FILL);
  if (!small.detections.some((d) => d.confidence >= SHOW_CONFIDENCE)) return { ...full, rescaled: false };
  return { ...small, ms: full.ms + small.ms, rescaled: true };
}

export function parseYoloLabels(text: string, width: number, height: number, classes: string[]) {
  return text
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [c, xc, yc, w, h] = line.split(/\s+/).map(Number);
      return {
        classIndex: c,
        label: classes[c],
        x1: (xc - w / 2) * width,
        y1: (yc - h / 2) * height,
        x2: (xc + w / 2) * width,
        y2: (yc + h / 2) * height,
      };
    });
}
