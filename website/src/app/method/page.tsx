import type { Metadata } from "next";
import { Figure, Section } from "@/components/ui";
import { metrics, samples } from "@/lib/data";

export const metadata: Metadata = { title: "Method" };

const STEPS = [
  { t: "Collect", d: "Each student photographs one object: 95 object + 5 background photos, uploaded to the shared Drive." },
  { t: "Clean", d: "Fix sizes, names and EXIF rotation; 224×224 RGB; remove near-duplicate photos." },
  { t: "Split", d: "Stratified 70 / 15 / 15 per class. Background photos are split the same way." },
  { t: "Compose", d: "Paste 2–6 photos from one split onto a background from that split: scatter, grid or collage." },
  { t: "Check", d: "No photo in two splits (path + image fingerprint); every label validated." },
  { t: "Train", d: "Fine-tune ImageNet EfficientNet-B0 (single object) and COCO YOLOv8s (multi-object)." },
  { t: "Evaluate", d: "Score once on the held-out test split; export to ONNX for this site." },
];

const HYPER = [
  ["Starting weights", "yolov8s.pt (COCO-pretrained, 80 → 73 classes)"],
  ["Parameters", "11.2 M"],
  ["Input size", "640 × 640"],
  ["Epochs", "60 (early-stopping patience 15)"],
  ["Batch size", "16"],
  ["Optimizer", "AdamW, lr 1.3e-4 (auto), 3 warm-up epochs"],
  ["Loss weights", "box 7.5 · class 0.5 · DFL 1.5"],
  ["Augmentation", "mosaic (off last 10 epochs), HSV (h 0.015, s 0.7, v 0.4), flip 0.5, scale 0.5, translate 0.1, erasing 0.4"],
  ["Prediction", "confidence ≥ 0.5, class-agnostic NMS at IoU 0.7"],
];

export default function MethodPage() {
  const d = metrics.dataset;
  const leak = d.leakage_check as Record<string, number>;
  const example = samples.composites.find((c) => c.layout === "collage") ?? samples.composites[0];
  return (
    <div className="mx-auto max-w-6xl px-4 pt-12 sm:px-6">
      <p className="text-sm font-medium text-brand">How it was built</p>
      <h1 className="mt-2 text-4xl font-semibold tracking-tight">Method</h1>

      <Section title="Pipeline">
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
          {STEPS.map((s, i) => (
            <li key={s.t} className="relative rounded-xl border border-line bg-surface p-4">
              <span className="tabular text-xs font-semibold text-brand">{String(i + 1).padStart(2, "0")}</span>
              <p className="mt-1 font-semibold">{s.t}</p>
              <p className="mt-1 text-xs leading-relaxed text-ink-2">{s.d}</p>
            </li>
          ))}
        </ol>
      </Section>

      <Section title="Multi-object dataset">
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure title="Example test image and its YOLO label file" caption="One line per object: class index, box centre x, centre y, width, height (normalized 0–1). Labels are written automatically from where each photo is pasted.">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={example.src} alt="Generated multi-object test image" className="w-full rounded-md" />
            <pre className="mt-3 overflow-x-auto rounded-md bg-surface-2 p-3 text-xs leading-relaxed">{example.labels.trim()}</pre>
          </Figure>
          <div className="space-y-6">
            <Figure title="Dataset size">
              <table className="tabular w-full text-sm">
                <thead className="text-left text-xs text-ink-2">
                  <tr><th className="py-1 font-medium">Split</th><th className="py-1 font-medium">Images</th><th className="py-1 font-medium">Objects</th><th className="py-1 font-medium">Min. per class</th></tr>
                </thead>
                <tbody>
                  {["train", "val", "test"].map((s) => (
                    <tr key={s} className="border-t border-line">
                      <td className="py-2 font-medium capitalize">{s}</td>
                      <td className="py-2">{d.images[s]?.toLocaleString()}</td>
                      <td className="py-2">{d.objects[s]?.toLocaleString()}</td>
                      <td className="py-2">{d.min_instances_per_class[s]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 text-xs text-ink-2">
                Layouts: {Object.entries(d.layouts).map(([k, v]) => `${k} ${v}`).join(" · ")}
              </p>
            </Figure>
            <Figure title="Split and label checks" caption="Run by scripts/01_make_multi_object.py every time the dataset is generated; it stops with an error if any check fails.">
              <ul className="space-y-1.5 text-sm">
                <li className="flex justify-between"><span>Source photos in more than one split (path)</span><span className="tabular font-semibold">{leak["source_paths_in_2+_splits"]}</span></li>
                <li className="flex justify-between"><span>Source photos in more than one split (fingerprint)</span><span className="tabular font-semibold">{leak["source_fingerprints_in_2+_splits"]}</span></li>
                <li className="flex justify-between"><span>Background photos in more than one split</span><span className="tabular font-semibold">{leak["background_fingerprints_in_2+_splits"]}</span></li>
                <li className="flex justify-between"><span>Labels checked / invalid</span><span className="tabular font-semibold">{d.label_validation.boxes_checked.toLocaleString()} / {d.label_validation.invalid}</span></li>
              </ul>
            </Figure>
          </div>
        </div>
      </Section>

      <Section title="Detector training (transfer learning)">
        <Figure title="YOLOv8s configuration">
          <table className="w-full text-sm">
            <tbody>
              {HYPER.map(([k, v]) => (
                <tr key={k} className="border-t border-line first:border-t-0">
                  <td className="w-48 py-2 pr-4 align-top font-medium">{k}</td>
                  <td className="py-2 text-ink-2">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Figure>
      </Section>

      <Section title="Running in the browser">
        <p className="max-w-3xl text-ink-2">
          Both trained models were exported to ONNX and checked against PyTorch on test images (identical boxes and top-1 predictions). This site runs them with ONNX Runtime Web on your GPU through WebGPU, or on the CPU through WebAssembly, so no server is involved and uploaded photos never leave your device.
        </p>
      </Section>
    </div>
  );
}
