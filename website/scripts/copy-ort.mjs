// Copy the onnxruntime-web runtime into public/ort so the models run without a CDN.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const src = "node_modules/onnxruntime-web/dist";
const dst = "public/ort";
mkdirSync(dst, { recursive: true });
for (const f of readdirSync(src)) {
  if (/^ort\.webgpu\.min\.js$|^ort-wasm-simd-threaded\.asyncify\.(mjs|wasm)$/.test(f)) cpSync(`${src}/${f}`, `${dst}/${f}`);
}
console.log("copied onnxruntime-web to", dst);
