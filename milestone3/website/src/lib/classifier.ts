import type { InferenceSession } from "onnxruntime-web";
import { createSession, loadOrt, type Backend } from "./ort";

const SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export type Prediction = { label: string; probability: number };
export type ClassifyResult = { top: Prediction[]; ms: number; backend: Backend };

let loaded: Promise<{ session: InferenceSession; backend: Backend }> | null = null;

export function loadClassifier(onProgress?: (f: number) => void) {
  loaded ??= createSession("/models/classifier.onnx", [1, 3, SIZE, SIZE], onProgress).catch((e) => {
    loaded = null;
    throw e;
  });
  return loaded;
}

export async function classify(source: CanvasImageSource, classes: string[], k = 5): Promise<ClassifyResult> {
  const [{ session, backend }, ort] = await Promise.all([loadClassifier(), loadOrt()]);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, SIZE, SIZE);
  const { data: rgba } = ctx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) data[c * plane + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  }
  const t0 = performance.now();
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]) });
  const ms = performance.now() - t0;
  const logits = Array.from(out[session.outputNames[0]].data as Float32Array);
  const max = Math.max(...logits);
  const exp = logits.map((l) => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  const top = exp
    .map((e, i) => ({ label: classes[i], probability: e / sum }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, k);
  return { top, ms, backend };
}
