import type { Metadata } from "next";
import { Reveal } from "@/components/motion";
import { Figure, PageHeader, Section } from "@/components/ui";
import { GRIDS, metrics, samples } from "@/lib/data";

export const metadata: Metadata = { title: "Method" };

const STEPS = [
  { t: "Collect", d: "Each student photographs one object and uploads the photos to the shared Drive." },
  { t: "Clean", d: "Fix sizes, names and EXIF rotation; 224×224 RGB; remove exact and near-duplicate photos before splitting." },
  { t: "Split", d: "Stratified 70 / 15 / 15 per class. OBJ054 is excluded because its photos do not show the object." },
  { t: "Label", d: "Grounding DINO and OWLv2 each box the object in every photo. A box is kept only when the two agree (IoU ≥ 0.5); it is their mean." },
  { t: "Compose", d: "Concatenate photos from one split into 2×2, 3×3, 4×4, 5×4 or 5×5 grids: 4–25 distinct objects, mild colour and flip changes per photo." },
  { t: "Train", d: "Fine-tune ImageNet EfficientNet-B0 (single object) and COCO YOLOv8s at 1120 × 1120 (multi-object)." },
  { t: "Evaluate", d: "Score once on the held-out test grids; export to ONNX for this site." },
];

const HYPER = [
  ["Starting weights", "yolov8s.pt (COCO-pretrained, 80 → 72 classes)"],
  ["Parameters", "11.2 M"],
  ["Input size", "1120 × 1120 (largest grid, 5×5, at full resolution)"],
  ["Epochs", "60 (early-stopping patience 15), Colab T4 GPU"],
  ["Batch size", "8"],
  ["Optimizer", "AdamW, lr 1.3e-4 (auto), 3 warm-up epochs"],
  ["Loss weights", "box 7.5 · class 0.5 · DFL 1.5"],
  ["Augmentation", "mosaic (off last 10 epochs), HSV (h 0.015, s 0.7, v 0.4), flip 0.5, scale 0.5, translate 0.1, erasing 0.4"],
  ["Prediction", "confidence ≥ 0.5, class-agnostic NMS at IoU 0.7"],
];

export default function MethodPage() {
  const d = metrics.dataset;
  const leak = d.leakage_check;
  const example = samples.composites.find((c) => c.grid === "3x3") ?? samples.composites[0];
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <PageHeader kicker="How it was built" title={<>From photos to a <span className="font-serif font-normal italic text-gradient">live detector</span></>} />

      <Section title="Pipeline">
        <ol className="relative grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
          <span aria-hidden className="pointer-events-none absolute inset-x-6 top-[1.6rem] hidden h-px bg-gradient-to-r from-brand via-brand-2 to-brand opacity-50 lg:block" />
          {STEPS.map((s, i) => (
            <Reveal as="li" key={s.t} delay={i * 0.07} y={24} className="relative rounded-xl border border-line bg-surface p-4 transition-[transform,border-color] duration-300 hover:-translate-y-1 hover:border-line-strong">
              <span className="tabular relative inline-grid h-7 w-7 place-items-center rounded-full border border-line-strong bg-surface-2 text-[11px] font-semibold text-brand">{String(i + 1).padStart(2, "0")}</span>
              <p className="mt-1 font-semibold">{s.t}</p>
              <p className="mt-1 text-xs leading-relaxed text-ink-2">{s.d}</p>
            </Reveal>
          ))}
        </ol>
      </Section>

      <Section title="Multi-object dataset">
        <div className="grid gap-6 lg:grid-cols-2">
          <Figure title="Example test image and its YOLO label file" caption="One line per object: class index, box centre x, centre y, width, height (normalized 0–1 to the image size). Each box is the object's cross-checked box, moved to its cell in the grid.">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={example.src} alt="3×3 multi-object test grid" className="w-full rounded-md" />
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
                Grids per split: {GRIDS.map((g) => `${g.replace("x", "×")}: ${d.grids.train[g] ?? 0} / ${d.grids.val[g] ?? 0} / ${d.grids.test[g] ?? 0}`).join(" · ")} (train / val / test)
              </p>
            </Figure>
            <Figure title="Split and label checks" caption="Run by scripts/02_make_grids.py every time the dataset is generated; it stops with an error if any check fails.">
              <ul className="space-y-1.5 text-sm">
                <li className="flex justify-between"><span>Source photos in more than one split (path)</span><span className="tabular font-semibold">{leak["source_paths_in_2+_splits"]}</span></li>
                <li className="flex justify-between"><span>Source photos in more than one split (fingerprint)</span><span className="tabular font-semibold">{leak["source_fingerprints_in_2+_splits"]}</span></li>
                <li className="flex justify-between"><span>Unique source photos (train / val / test)</span><span className="tabular font-semibold">{["train", "val", "test"].map((k) => leak.unique_source_images[k]?.toLocaleString()).join(" / ")}</span></li>
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
