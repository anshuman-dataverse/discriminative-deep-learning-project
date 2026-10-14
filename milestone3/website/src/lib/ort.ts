import type * as OrtTypes from "onnxruntime-web";

export type Ort = typeof OrtTypes;
export type Backend = "webgpu" | "wasm";

declare global {
  interface Window {
    ort?: Ort;
  }
}

let ortPromise: Promise<Ort> | null = null;

export function loadOrt(): Promise<Ort> {
  if (ortPromise) return ortPromise;
  ortPromise = new Promise<Ort>((resolve, reject) => {
    if (window.ort) return resolve(window.ort);
    const s = document.createElement("script");
    s.src = "/ort/ort.webgpu.min.js";
    s.async = true;
    s.onload = () => {
      const ort = window.ort!;
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = window.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
      resolve(ort);
    };
    s.onerror = () => {
      ortPromise = null;
      reject(new Error("Could not load the ONNX runtime"));
    };
    document.head.appendChild(s);
  });
  return ortPromise;
}

export function preferredBackend(): Backend {
  if (typeof window === "undefined") return "wasm";
  if (new URLSearchParams(window.location.search).get("backend") === "wasm") return "wasm";
  return "gpu" in navigator ? "webgpu" : "wasm";
}

const GPU_TIMEOUT_MS = 20000;

function within<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => reject(new Error("timed out")), ms);
    p.then(
      (v) => {
        clearTimeout(id);
        resolve(v);
      },
      (e) => {
        clearTimeout(id);
        reject(e);
      },
    );
  });
}

async function open(ort: Ort, bytes: Uint8Array, backend: Backend, inputShape: number[]) {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: [backend] });
  const zeros = new ort.Tensor("float32", new Float32Array(inputShape.reduce((a, b) => a * b, 1)), inputShape);
  await session.run({ [session.inputNames[0]]: zeros });
  return session;
}

export async function createSession(
  url: string,
  inputShape: number[],
  onProgress?: (fraction: number) => void,
): Promise<{ session: OrtTypes.InferenceSession; backend: Backend }> {
  const ort = await loadOrt();
  const bytes = await fetchWithProgress(url, onProgress);
  if (preferredBackend() === "webgpu") {
    try {
      return { session: await within(open(ort, bytes, "webgpu", inputShape), GPU_TIMEOUT_MS), backend: "webgpu" };
    } catch {
      onProgress?.(1);
    }
  }
  return { session: await open(ort, bytes, "wasm", inputShape), backend: "wasm" };
}

async function fetchWithProgress(url: string, onProgress?: (f: number) => void): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download ${url} (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress?.(received / total);
  }
  onProgress?.(1);
  const out = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}
