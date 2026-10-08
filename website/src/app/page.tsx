import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { Playground } from "@/components/playground/playground";
import { m1, metrics, pct } from "@/lib/data";

export default function Home() {
  const acc = metrics["at_conf_0.5"];
  const best = m1.models.find((m) => m.model === m1.best_model)!;
  const kpis = [
    { value: metrics.mAP50.toFixed(3), label: "Detector mAP@0.5", sub: `${metrics.test_images} held-out test images` },
    { value: pct(acc.object_accuracy), label: "Objects with correct ID + box", sub: `${acc.objects_correct_id_and_location} of ${acc.objects}` },
    { value: pct(best.test_acc, 2), label: "Single-object ID accuracy", sub: `EfficientNet-B0 · ${m1.num_classes} classes` },
    { value: "73", label: "Object classes", sub: "final Object IDs, whole class" },
  ];
  return (
    <>
      <section className="relative overflow-hidden border-b border-line">
        <div className="dot-grid absolute inset-0 opacity-60 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]" />
        <div className="relative mx-auto max-w-6xl px-4 pb-12 pt-14 sm:px-6 sm:pt-20">
          <p className="text-sm font-medium text-brand">IE 7615 · Discriminative Deep Learning · Group 1</p>
          <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight sm:text-5xl">
            Identify one object. Detect, identify and locate many.
          </h1>
          <p className="mt-4 max-w-2xl text-lg text-ink-2">
            Two deep learning models trained on 7,300 photos of 73 everyday objects shot by our class. Try them live below: everything runs in your browser.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <a href="#demo" className="inline-flex items-center gap-2 rounded-lg bg-ink px-4 py-2.5 text-sm font-medium text-page hover:opacity-90">
              Try the models <ArrowRight className="h-4 w-4" />
            </a>
            <Link href="/metrics" className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-4 py-2.5 text-sm font-medium hover:bg-surface-2">
              See the metrics
            </Link>
          </div>
          <dl className="mt-10 grid grid-cols-2 gap-3 lg:grid-cols-4">
            {kpis.map((k) => (
              <div key={k.label} className="rounded-xl border border-line bg-surface p-4">
                <dt className="text-xs text-ink-2">{k.label}</dt>
                <dd className="mt-1 text-3xl font-semibold tracking-tight">{k.value}</dd>
                <dd className="mt-1 text-xs text-muted">{k.sub}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-4 pt-10 sm:px-6">
        <div className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Live demo</h2>
          <p className="mt-1 text-ink-2">
            The first run downloads the model once; after that each image takes milliseconds. Test images were never seen in training.
          </p>
        </div>
        <Playground />

        <section className="mt-16 grid gap-4 sm:grid-cols-3">
          {[
            { n: "1", t: "Single-object ID", d: "Four CNNs compared on a stratified split; ImageNet-pretrained EfficientNet-B0 fine-tuned to tell the 73 objects apart." },
            { n: "2", t: "Multi-object images", d: "Test-split photos pasted onto background photos in scatter, grid and collage layouts, with boxes written automatically. No photo is in two splits." },
            { n: "3", t: "Detection & location", d: "COCO-pretrained YOLOv8s fine-tuned with transfer learning; it returns the ID, confidence and box of every object." },
          ].map((s) => (
            <div key={s.n} className="rounded-xl border border-line bg-surface p-5">
              <span className="tabular text-sm font-semibold text-brand">0{s.n}</span>
              <h3 className="mt-1 font-semibold">{s.t}</h3>
              <p className="mt-2 text-sm text-ink-2">{s.d}</p>
            </div>
          ))}
        </section>
        <div className="mt-6 flex gap-4 text-sm">
          <Link href="/method" className="font-medium text-ink underline-offset-4 hover:underline">How it was built →</Link>
          <Link href="/objects" className="font-medium text-ink underline-offset-4 hover:underline">Browse all 73 objects →</Link>
        </div>
      </div>
    </>
  );
}
