import type { Metadata } from "next";
import { Bars, EpochLines } from "@/components/charts";
import { Figure, PageHeader, Section, StatTile } from "@/components/ui";
import { m1, metrics, pct, training } from "@/lib/data";

export const metadata: Metadata = { title: "Metrics" };

const MODEL_NAMES: Record<string, string> = {
  EfficientNetB0: "EfficientNet-B0",
  MobileNetV3: "MobileNetV3-Large",
  ResNet18: "ResNet18",
  SimpleCNN: "SimpleCNN (scratch)",
};

export default function MetricsPage() {
  const acc = metrics["at_conf_0.5"];
  const epochs = training.map((r) => ({
    epoch: r.epoch,
    map50: r["metrics/mAP50(B)"],
    map5095: r["metrics/mAP50-95(B)"],
    trainCls: r["train/cls_loss"],
    valCls: r["val/cls_loss"],
    trainBox: r["train/box_loss"],
    valBox: r["val/box_loss"],
  }));
  const perClass = [...metrics.per_class].sort((a, b) => a.AP50 - b.AP50);
  const pairs = Object.entries(metrics.errors.wrong_id_pairs).slice(0, 8);
  const missed = Object.entries(metrics.errors.missed_by_class).slice(0, 8);
  const layouts = Object.entries(metrics.by_layout);
  const best = m1.models.find((m) => m.model === m1.best_model)!;
  const hardest = Object.entries(m1.per_class_f1).sort((a, b) => a[1] - b[1]).slice(0, 5);

  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <PageHeader kicker="Results" title={<>Measured on <span className="font-serif font-normal italic text-gradient">held-out</span> data</>}>
        Every number on this page is computed on held-out test data that played no part in training or model selection.
      </PageHeader>

      <Section title="Multi-object detection (YOLOv8s)" kicker="Milestone 2" id="detector">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
          <StatTile label="mAP@0.5" value={metrics.mAP50.toFixed(3)} />
          <StatTile label="mAP@0.5:0.95" value={metrics["mAP50-95"].toFixed(3)} />
          <StatTile label="Precision / Recall" value={`${metrics.precision.toFixed(2)} / ${metrics.recall.toFixed(2)}`} />
          <StatTile label="Objects correct (ID + box)" value={pct(acc.object_accuracy)} sub={`${acc.objects_correct_id_and_location} of ${acc.objects}, conf ≥ 0.5`} />
          <StatTile label="Images fully correct" value={pct(acc.image_accuracy)} sub={`${acc.images_fully_correct} of ${acc.images}`} />
        </div>
        <p className="mt-3 text-sm text-ink-2">
          {metrics.test_images} test images built only from test-split photos · {metrics.inference_ms_per_image} ms per image on an Apple M4 GPU
          {metrics.best_f1 && ` · best F1 ${metrics.best_f1.f1.toFixed(3)} at confidence ${metrics.best_f1.confidence}`}.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <Figure title="Validation mAP per epoch" caption="Both mAP measures rise together; the gap is small because predicted boxes are very tight.">
            <EpochLines
              data={epochs}
              domain={[0, 1]}
              series={[
                { key: "map50", label: "mAP@0.5", color: "var(--series-1)" },
                { key: "map5095", label: "mAP@0.5:0.95", color: "var(--series-2)" },
              ]}
            />
          </Figure>
          <Figure title="Classification loss per epoch" caption="Validation loss keeps falling with training loss, so the model is not overfitting. The step near the end is where mosaic augmentation switches off.">
            <EpochLines
              data={epochs}
              format="fixed2"
              series={[
                { key: "trainCls", label: "Train", color: "var(--series-1)" },
                { key: "valCls", label: "Validation", color: "var(--series-2)" },
              ]}
            />
          </Figure>
        </div>

        <Figure
          className="mt-6"
          title="Test AP@0.5 for each of the 73 objects"
          caption={`Sorted from hardest to easiest. Orange bars are below 0.80. ${metrics.class_tiers["excellent (AP50 >= 0.95)"]?.count ?? 0} of 73 classes reach 0.95 or more.`}
        >
          <Bars data={perClass} xKey="object_id" yKey="AP50" label="AP@0.5" height={260} showTicks={false} highlightBelow={0.8} />
          <details className="mt-3 text-sm">
            <summary className="cursor-pointer text-ink-2">Show as table</summary>
            <div className="mt-2 max-h-72 overflow-y-auto rounded-lg border border-line">
              <table className="tabular w-full text-xs">
                <thead className="sticky top-0 bg-surface-2 text-left text-ink-2">
                  <tr>{["Object ID", "Precision", "Recall", "AP@0.5", "AP@0.5:0.95"].map((h) => <th key={h} className="px-3 py-1.5 font-medium">{h}</th>)}</tr>
                </thead>
                <tbody>
                  {perClass.map((r) => (
                    <tr key={r.object_id} className="border-t border-line">
                      <td className="px-3 py-1.5 font-medium">{r.object_id}</td>
                      <td className="px-3 py-1.5">{r.precision.toFixed(3)}</td>
                      <td className="px-3 py-1.5">{r.recall.toFixed(3)}</td>
                      <td className="px-3 py-1.5">{r.AP50.toFixed(3)}</td>
                      <td className="px-3 py-1.5">{r["AP50-95"].toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </Figure>

        <div className="mt-6 grid gap-6 lg:grid-cols-3">
          <Figure title="Accuracy by layout" caption="Share of objects with the correct ID and box, and of images where everything is correct.">
            <table className="tabular w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr><th className="py-1 font-medium">Layout</th><th className="py-1 font-medium">Images</th><th className="py-1 font-medium">Object acc.</th><th className="py-1 font-medium">Image acc.</th></tr>
              </thead>
              <tbody>
                {layouts.map(([k, v]) => (
                  <tr key={k} className="border-t border-line">
                    <td className="py-2 font-medium capitalize">{k}</td>
                    <td className="py-2 text-ink-2">{v.images}</td>
                    <td className="py-2">{pct(v.object_accuracy)}</td>
                    <td className="py-2">{pct(v.image_accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Figure>
          <Figure title="Class tiers" caption="Classes grouped by test AP@0.5.">
            <ul className="space-y-3 text-sm">
              {Object.entries(metrics.class_tiers).map(([tier, t]) => (
                <li key={tier}>
                  <p className="flex justify-between font-medium capitalize">
                    <span>{tier}</span>
                    <span className="tabular text-ink-2">{t.count} classes{t.mean_AP50 !== null && ` · mean ${t.mean_AP50.toFixed(3)}`}</span>
                  </p>
                  {t.count > 0 && t.count < 20 && <p className="mt-0.5 text-xs text-ink-2">{t.classes.join(", ")}</p>}
                </li>
              ))}
            </ul>
          </Figure>
          <Figure title="Where it goes wrong" caption="Most frequent wrong IDs (true → predicted) and missed objects at conf ≥ 0.5.">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <ul className="tabular space-y-1">
                {pairs.map(([p, n]) => (
                  <li key={p} className="flex justify-between gap-2"><span>{p}</span><span className="text-ink-2">{n}</span></li>
                ))}
              </ul>
              <ul className="tabular space-y-1">
                {missed.map(([c, n]) => (
                  <li key={c} className="flex justify-between gap-2"><span>{c} missed</span><span className="text-ink-2">{n}</span></li>
                ))}
              </ul>
            </div>
          </Figure>
        </div>

        <div className="mt-6 grid gap-6 md:grid-cols-3">
          {[
            { src: "/figures/test_BoxPR_curve.png", t: "Precision-recall curve" },
            { src: "/figures/test_BoxF1_curve.png", t: "F1 vs confidence" },
            { src: "/figures/test_confusion_matrix_normalized.png", t: "Confusion matrix (normalized)" },
          ].map((f) => (
            <Figure key={f.src} title={f.t} caption="Generated by Ultralytics on the test split.">
              <a href={f.src} target="_blank" rel="noreferrer">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={f.src} alt={f.t} loading="lazy" className="w-full rounded-md bg-white" />
              </a>
            </Figure>
          ))}
        </div>
      </Section>

      <Section title="Single-object identification (CNN comparison)" kicker="Milestone 1" id="classifier">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <StatTile label="Best model" value="EfficientNet-B0" />
          <StatTile label="Test accuracy" value={pct(best.test_acc, 2)} />
          <StatTile label="Macro F1" value={best.test_macro_f1.toFixed(3)} />
          <StatTile label="Classes" value={String(m1.num_classes)} />
        </div>
        <div className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <Figure title="Test accuracy by model" caption="Three ImageNet-pretrained networks against a CNN trained from scratch, on the same split.">
            <Bars
              data={m1.models.map((m) => ({ model: MODEL_NAMES[m.model] ?? m.model, acc: m.test_acc }))}
              xKey="model"
              yKey="acc"
              label="Test accuracy"
              format="pct"
              height={240}
              gap="28%"
            />
          </Figure>
          <Figure title="Model comparison" caption="Inference time is per image at batch size 1.">
            <table className="tabular w-full text-sm">
              <thead className="text-left text-xs text-ink-2">
                <tr>{["Model", "Test acc.", "F1", "Params", "ms"].map((h) => <th key={h} className="py-1 font-medium">{h}</th>)}</tr>
              </thead>
              <tbody>
                {m1.models.map((m) => (
                  <tr key={m.model} className="border-t border-line">
                    <td className="py-2 font-medium">{MODEL_NAMES[m.model] ?? m.model}</td>
                    <td className="py-2">{pct(m.test_acc, 2)}</td>
                    <td className="py-2">{m.test_macro_f1.toFixed(3)}</td>
                    <td className="py-2 text-ink-2">{m.params_M}M</td>
                    <td className="py-2 text-ink-2">{m.inference_ms_per_img}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-4 text-xs text-ink-2">
              Hardest classes for EfficientNet-B0 (F1): {hardest.map(([c, f]) => `${c} ${f.toFixed(2)}`).join(", ")}.
            </p>
          </Figure>
        </div>
      </Section>
    </div>
  );
}
