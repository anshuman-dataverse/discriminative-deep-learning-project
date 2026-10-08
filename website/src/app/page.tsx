import { ArrowRight, Boxes, Cpu, Database, Layers, ShieldCheck, Sparkles } from "lucide-react";
import Link from "next/link";
import { HeroVisual } from "@/components/hero-visual";
import { Bar, CountUp, HeroScroll, Reveal, Tilt } from "@/components/motion";
import { ObjectMarquee } from "@/components/object-marquee";
import { Playground } from "@/components/playground/playground";
import { ScrollStory, type Step } from "@/components/scroll-story";
import { m1, metrics, pct } from "@/lib/data";

const LAYOUT_NAMES: Record<string, string> = { scatter: "Scatter", grid: "Grid", collage: "Collage", large: "Large" };

export default function Home() {
  const acc = metrics["at_conf_0.5"];
  const best = m1.models.find((m) => m.model === m1.best_model)!;
  const d = metrics.dataset;
  const totalImages = Object.values(d.images).reduce((a, b) => a + b, 0);
  const stats = [
    { value: metrics.mAP50.toFixed(3), label: "detector mAP@0.5" },
    { value: pct(best.test_acc, 2), label: "single-object accuracy" },
    { value: pct(acc.object_accuracy), label: "objects with correct ID + box" },
    { value: String(acc.false_alarms), label: "false alarms on test" },
  ];
  const layouts = Object.entries(metrics.by_layout).sort((a, b) => b[1].object_accuracy - a[1].object_accuracy);
  const models = [...m1.models].sort((a, b) => b.test_acc - a.test_acc);
  const steps: Step[] = [
    { kicker: "Collect", title: "One object per student", body: "Every student photographed one everyday object. After fixing sizes and rotation and removing duplicates, each class was split 70 / 15 / 15.", stat: (6942).toLocaleString(), statLabel: "clean photos of 73 objects" },
    { kicker: "Compose", title: "Paste, and the labels write themselves", body: "Photos from one split are pasted onto a background from the same split. Each paste position becomes a line in the YOLO label file.", stat: totalImages.toLocaleString(), statLabel: "multi-object images" },
    { kicker: "Detect", title: "One look at the whole scene", body: "YOLOv8s, fine-tuned from COCO, reads the full image in a single pass and draws a box around every object it finds.", stat: metrics.mAP50.toFixed(3), statLabel: "test mAP@0.5" },
    { kicker: "Identify", title: "Every box gets its Object ID", body: `Each box is assigned one of 73 Object IDs. On ${metrics.test_images} held-out test scenes the detector made ${acc.false_alarms} false alarms.`, stat: pct(acc.object_accuracy), statLabel: "objects with correct ID + box" },
  ];

  return (
    <>
      <section className="relative -mt-16 overflow-hidden pt-16">
        <div className="aurora pointer-events-none absolute inset-0" />
        <div className="dot-grid pointer-events-none absolute inset-0 opacity-40 [mask-image:radial-gradient(ellipse_at_70%_30%,black,transparent_70%)]" />
        <HeroScroll
          className="relative mx-auto grid max-w-6xl items-center gap-6 px-4 pb-10 pt-14 sm:px-6 lg:min-h-[640px] lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] lg:pt-10"
          text={
            <div className="fade-up">
              <span className="inline-flex items-center gap-2 rounded-full border border-line bg-surface/60 px-3 py-1 text-xs text-ink-2">
                <span className="h-1.5 w-1.5 rounded-full bg-good shadow-[0_0_10px_var(--good)]" />
                Live demo · runs in your browser
              </span>
              <h1 className="mt-5 text-[2.6rem] font-semibold leading-[1.04] tracking-[-0.035em] sm:text-6xl lg:text-[3.15rem] xl:text-[3.5rem]">
                <span className="block sm:whitespace-nowrap">Identify one object.</span>
                <span className="block font-serif text-[1.08em] font-normal italic tracking-[-0.01em] text-gradient">Locate them all.</span>
              </h1>
              <p className="mt-5 max-w-md text-lg leading-relaxed text-ink-2">
                Two deep learning models trained on {(6942).toLocaleString()} photos of 73 everyday objects from our class. Pick a test image, upload a photo, or use your camera.
              </p>
              <div className="mt-7 flex flex-wrap gap-3">
                <a href="#demo" className="group inline-flex items-center gap-2 rounded-xl bg-ink px-4 py-2.5 text-sm font-medium text-page shadow-[0_10px_30px_var(--glow)] hover:opacity-90">
                  Open the demo <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                </a>
                <Link href="/metrics" className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface/60 px-4 py-2.5 text-sm font-medium hover:border-line-strong">
                  View results
                </Link>
              </div>
              <dl className="mt-10 grid max-w-md grid-cols-2 gap-x-6 gap-y-5">
                {stats.map((s) => (
                  <div key={s.label}>
                    <dt className="tabular text-2xl font-semibold tracking-tight"><CountUp value={s.value} /></dt>
                    <dd className="mt-0.5 text-xs text-muted">{s.label}</dd>
                  </div>
                ))}
              </dl>
            </div>
          }
          visual={
            <div className="fade-up relative h-[380px] sm:h-[460px] lg:-mr-16 lg:h-[600px] [mask-image:linear-gradient(to_right,transparent,black_14%,black_86%,transparent)]">
              <HeroVisual />
              <p className="absolute bottom-2 left-1/2 -translate-x-1/2 whitespace-nowrap text-[11px] text-muted">
                Held-out test images with their ground-truth labels
              </p>
            </div>
          }
        />
      </section>

      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="flex flex-wrap items-center justify-center gap-x-8 gap-y-2 border-y border-line py-4 text-xs text-muted">
          <span>ImageNet → EfficientNet-B0</span>
          <span>COCO → YOLOv8s</span>
          <span>PyTorch → ONNX</span>
          <span>ONNX Runtime Web · WebGPU / WebAssembly</span>
        </div>
      </div>

      <section className="pt-24">
        <Reveal className="mx-auto max-w-6xl px-4 text-center sm:px-6">
          <p className="text-sm font-medium text-brand">The dataset</p>
          <h2 className="mx-auto mt-1 max-w-3xl text-4xl font-semibold tracking-[-0.03em] sm:text-6xl">
            <span className="font-serif font-normal italic text-gradient">73</span> objects, photographed by our class
          </h2>
        </Reveal>
        <Reveal delay={0.1} y={40}>
          <ObjectMarquee />
        </Reveal>
      </section>

      <ScrollStory steps={steps} />

      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <section id="demo" className="scroll-mt-24 pt-12">
          <Reveal className="mb-6 max-w-2xl">
            <p className="text-sm font-medium text-brand">Studio</p>
            <h2 className="mt-1 text-3xl font-semibold tracking-tight sm:text-5xl">Try both models</h2>
            <p className="mt-3 text-ink-2">The first run downloads the model once. Test images were never seen in training, and they are scored against their true labels.</p>
          </Reveal>
          <Reveal y={60}>
            <Playground />
          </Reveal>
        </section>

        <section className="pt-28">
          <Reveal className="mb-8 max-w-2xl">
            <p className="text-sm font-medium text-brand">Under the hood</p>
            <h2 className="mt-1 text-3xl font-semibold tracking-tight sm:text-5xl">Two models, one browser tab</h2>
          </Reveal>
          <div className="grid gap-4 md:grid-cols-6">
            <Card delay={0} className="md:col-span-3" icon={Database} kicker="Data" title={`${(6942).toLocaleString()} clean photos, 73 objects`}>
              Every student photographed one object. We fixed sizes and rotation, removed {28} duplicates, mapped the final Object IDs, and split each class 70 / 15 / 15.
            </Card>
            <Card delay={0.08} className="md:col-span-3" icon={Layers} kicker="Scenes" title={`${totalImages.toLocaleString()} multi-object images`}>
              Photos from one split are pasted onto backgrounds from the same split, so no photo appears in two splits. Boxes are written from the paste positions.
              <LayoutStrip />
            </Card>
            <Card delay={0} className="md:col-span-2" icon={Sparkles} kicker="Single object" title="EfficientNet-B0">
              Fine-tuned from ImageNet. Best of four CNNs at {pct(best.test_acc, 2)} test accuracy.
            </Card>
            <Card delay={0.08} className="md:col-span-2" icon={Boxes} kicker="Many objects" title="YOLOv8s">
              Fine-tuned from COCO with a 73-class head. {pct(acc.object_accuracy)} of test objects get the right ID and box.
            </Card>
            <Card delay={0.16} className="md:col-span-2" icon={Cpu} kicker="On device" title="ONNX in the browser">
              Both models are checked against PyTorch, then run on your GPU with WebGPU, or your CPU with WebAssembly.
            </Card>
          </div>
        </section>

        <section className="pt-28">
          <Reveal className="mb-8 flex flex-wrap items-end justify-between gap-4">
            <div className="max-w-2xl">
              <p className="text-sm font-medium text-brand">Results</p>
              <h2 className="mt-1 text-3xl font-semibold tracking-tight sm:text-5xl">Measured on held-out test data</h2>
            </div>
            <Link href="/metrics" className="inline-flex items-center gap-1.5 text-sm text-ink-2 hover:text-ink">
              All metrics <ArrowRight className="h-4 w-4" />
            </Link>
          </Reveal>
          <div className="grid gap-4 lg:grid-cols-2">
            <Reveal className="rounded-2xl border border-line bg-surface p-6">
              <p className="text-sm text-ink-2">Detector · {metrics.test_images} test images, {acc.objects} objects</p>
              <div className="mt-3 flex flex-wrap items-baseline gap-3">
                <CountUp value={metrics.mAP50.toFixed(3)} className="tabular text-gradient text-7xl font-semibold tracking-tight" />
                <span className="text-sm text-muted">mAP@0.5 · {metrics["mAP50-95"].toFixed(3)} at 0.5:0.95</span>
              </div>
              <p className="mt-6 text-xs font-medium uppercase tracking-wider text-muted">Objects correct by layout</p>
              <ul className="mt-3 space-y-3">
                {layouts.map(([k, v], i) => (
                  <li key={k} className="grid grid-cols-[5rem_1fr_3.5rem] items-center gap-3 text-sm">
                    <span className="text-ink-2">{LAYOUT_NAMES[k] ?? k}</span>
                    <span className="h-2 overflow-hidden rounded-full bg-surface-3">
                      <Bar value={v.object_accuracy} delay={i * 0.1} className="bg-gradient-to-r from-brand to-brand-2" />
                    </span>
                    <CountUp value={pct(v.object_accuracy)} className="tabular text-right" />
                  </li>
                ))}
              </ul>
            </Reveal>
            <Reveal delay={0.1} className="rounded-2xl border border-line bg-surface p-6">
              <p className="text-sm text-ink-2">Single-object classifier · {m1.num_classes} classes, 1,028 test photos</p>
              <ul className="mt-6 space-y-4">
                {models.map((m, i) => (
                  <li key={m.model}>
                    <div className="flex justify-between text-sm">
                      <span className={m.model === m1.best_model ? "font-medium" : "text-ink-2"}>{m.model.replace("EfficientNetB0", "EfficientNet-B0").replace("MobileNetV3", "MobileNetV3-Large").replace("SimpleCNN", "SimpleCNN (scratch)")}</span>
                      <CountUp value={pct(m.test_acc, 2)} className="tabular" />
                    </div>
                    <span className="mt-1.5 block h-2 overflow-hidden rounded-full bg-surface-3">
                      <Bar value={m.test_acc} delay={i * 0.1} className={m.model === m1.best_model ? "bg-gradient-to-r from-brand to-brand-2" : "bg-ink-2/40"} />
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-6 flex items-center gap-2 text-xs text-muted">
                <ShieldCheck className="h-3.5 w-3.5" /> Pretrained networks beat the scratch CNN by over 30 points.
              </p>
            </Reveal>
          </div>
        </section>

        <Reveal y={40} className="pt-28">
          <div className="shine relative overflow-hidden rounded-[28px] border border-line bg-surface px-6 py-16 text-center sm:py-20">
            <div className="aurora pointer-events-none absolute inset-0" />
            <div className="dot-grid pointer-events-none absolute inset-0 opacity-50 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
            <h2 className="relative mx-auto max-w-2xl text-4xl font-semibold tracking-[-0.03em] sm:text-6xl">
              See it find <span className="font-serif font-normal italic text-gradient">your</span> objects
            </h2>
            <p className="relative mx-auto mt-4 max-w-md text-ink-2">Upload a photo or point your camera at the desk. Everything runs on your device.</p>
            <div className="relative mt-8 flex flex-wrap justify-center gap-3">
              <a href="#demo" className="group inline-flex items-center gap-2 rounded-xl bg-ink px-5 py-3 text-sm font-medium text-page shadow-[0_10px_30px_var(--glow)] hover:opacity-90">
                Open the demo <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </a>
              <Link href="/objects" className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface/60 px-5 py-3 text-sm font-medium hover:border-line-strong">
                Explore the 73 objects
              </Link>
            </div>
          </div>
        </Reveal>
      </div>
    </>
  );
}

function Card({ icon: Icon, kicker, title, children, className = "", delay = 0 }: { icon: React.ComponentType<{ className?: string }>; kicker: string; title: string; children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <Reveal delay={delay} className={className}>
    <Tilt className="group h-full rounded-2xl">
    <div className="relative h-full overflow-hidden rounded-2xl border border-line bg-surface p-6 transition-colors group-hover:border-line-strong">
      <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-[radial-gradient(var(--glow),transparent_70%)] opacity-0 transition-opacity group-hover:opacity-100" />
      <span className="grid h-9 w-9 place-items-center rounded-xl border border-line bg-surface-2 text-brand">
        <Icon className="h-4 w-4" />
      </span>
      <p className="mt-4 text-xs font-medium uppercase tracking-wider text-muted">{kicker}</p>
      <h3 className="mt-1 text-lg font-semibold tracking-tight">{title}</h3>
      <div className="mt-2 text-sm leading-relaxed text-ink-2">{children}</div>
    </div>
    </Tilt>
    </Reveal>
  );
}

function LayoutStrip() {
  const shapes: { name: string; rects: [number, number, number, number][] }[] = [
    { name: "Scatter", rects: [[4, 6, 14, 14], [24, 4, 12, 12], [8, 24, 12, 12], [26, 22, 12, 14]] },
    { name: "Grid", rects: [[3, 3, 18, 18], [21, 3, 18, 18], [3, 21, 18, 18], [21, 21, 18, 18]] },
    { name: "Collage", rects: [[4, 5, 18, 18], [20, 8, 17, 17], [9, 21, 16, 16]] },
    { name: "Large", rects: [[3, 3, 36, 36]] },
  ];
  return (
    <div className="mt-4 flex gap-3">
      {shapes.map((s) => (
        <figure key={s.name} className="text-center">
          <svg viewBox="0 0 42 42" className="h-12 w-12 rounded-lg border border-line bg-surface-2">
            {s.rects.map(([x, y, w, h], i) => (
              <rect key={i} x={x} y={y} width={w} height={h} rx={1.5} fill="none" stroke={i === 0 ? "var(--brand)" : "var(--brand-2)"} strokeWidth={1.4} opacity={0.9} />
            ))}
          </svg>
          <figcaption className="mt-1 text-[10px] text-muted">{s.name}</figcaption>
        </figure>
      ))}
    </div>
  );
}
