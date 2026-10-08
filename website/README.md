# Demo website (Milestone 3)

Live: **https://ie7615-group1-object-detection.vercel.app**

A Next.js site that runs both trained models in the browser with ONNX Runtime Web (WebGPU when available, multi-threaded WebAssembly otherwise). Nothing is sent to a server.

- `/` – live demo: **Detect objects** (YOLOv8s, multi-object), **Identify object** (EfficientNet-B0, single object), **Build a scene**
- `/metrics` – test results of both milestones
- `/method` – how the data and models were built
- `/objects` – all 73 objects with per-class scores

## Data and models

`public/models/detector.onnx` and `public/models/classifier.onnx`, the sample images and `src/data/*.json` are written by `milestone2/scripts/05_export_web.py`, which also checks the ONNX models against PyTorch on test images.

## Run locally

```bash
npm install          # also copies the ONNX runtime into public/ort
npm run dev          # http://localhost:3000
npm run build        # production build
```

Code layout: `src/lib/detector.ts` (letterbox, decode, class-agnostic NMS), `src/lib/classifier.ts` (preprocessing, softmax), `src/lib/compose.ts` (scene layouts, ported from `milestone2/scripts/detector/compose.py`), `src/components/playground/` (the three demo modes).
