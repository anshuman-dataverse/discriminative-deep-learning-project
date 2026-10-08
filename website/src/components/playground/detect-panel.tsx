"use client";

import { CheckCircle2, CircleAlert, ScanSearch } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { classes, samples, thumb } from "@/lib/data";
import { detect, iou, loadDetector, parseYoloLabels, type Detection } from "@/lib/detector";
import { ImageInput, loadFromUrl, type LoadedImage } from "./image-input";
import { ImageStage, type StageBox } from "./image-stage";
import { ModelStatus } from "./model-status";
import { useModel } from "./use-model";

type Truth = ReturnType<typeof parseYoloLabels>;
const LAYOUTS = ["all", "scatter", "grid", "collage"] as const;

export function score(dets: Detection[], truth: Truth) {
  const free = new Set(truth.map((_, i) => i));
  const status = new Map<Detection, "correct" | "wrong">();
  for (const d of [...dets].sort((a, b) => b.confidence - a.confidence)) {
    let best = -1;
    let bestIou = 0.5;
    for (const j of free) {
      const v = iou(d, truth[j]);
      if (v >= bestIou) {
        best = j;
        bestIou = v;
      }
    }
    if (best < 0) {
      status.set(d, "wrong");
      continue;
    }
    free.delete(best);
    status.set(d, truth[best].classIndex === d.classIndex ? "correct" : "wrong");
  }
  return { status, missed: [...free].map((j) => truth[j]) };
}

export function DetectPanel() {
  const model = useModel(loadDetector);
  const [img, setImg] = useState<LoadedImage | null>(null);
  const [truth, setTruth] = useState<Truth | null>(null);
  const [all, setAll] = useState<Detection[]>([]);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [conf, setConf] = useState(0.5);
  const [showTruth, setShowTruth] = useState(true);
  const [layout, setLayout] = useState<(typeof LAYOUTS)[number]>("all");
  const [activeSample, setActiveSample] = useState<string | null>(null);

  const run = useCallback(
    async (input: LoadedImage, gt: Truth | null) => {
      setImg(input);
      setTruth(gt);
      setAll([]);
      setMs(null);
      setBusy(true);
      if (await model.ensure()) {
        const r = await detect(input.image, input.width, input.height, classes.detector, 0.1);
        setAll(r.detections);
        setMs(r.ms);
      }
      setBusy(false);
    },
    [model],
  );

  const shown = useMemo(() => all.filter((d) => d.confidence >= conf), [all, conf]);
  const scored = useMemo(() => (truth ? score(shown, truth) : null), [shown, truth]);
  const boxes: StageBox[] = [
    ...(truth && showTruth ? truth.map((t) => ({ ...t, truth: true })) : []),
    ...shown.map((d) => ({ ...d, status: scored?.status.get(d) })),
  ];
  const correct = scored ? [...scored.status.values()].filter((s) => s === "correct").length : 0;
  const composites = samples.composites.filter((c) => layout === "all" || c.layout === layout);
  const preview = samples.composites.find((c) => c.layout === "collage") ?? samples.composites[0];
  const runSample = async (c: (typeof samples.composites)[number]) => {
    setActiveSample(c.id);
    run(await loadFromUrl(c.src), parseYoloLabels(c.labels, 640, classes.detector));
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <div>
        <ImageInput
          onImage={(i) => {
            setActiveSample(null);
            run(i, null);
          }}
        >
          <ImageStage
            src={img?.src ?? null}
            width={img?.width ?? 640}
            height={img?.height ?? 640}
            boxes={boxes}
            busy={busy}
            empty={
              <button onClick={() => runSample(preview)} className="group relative h-full w-full" aria-label="Detect objects in a held-out test image">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview.src} alt="" className="h-full w-full object-cover opacity-45 transition group-hover:opacity-60" />
                <span className="absolute inset-0 grid place-items-center">
                  <span className="inline-flex items-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-medium text-page shadow-[0_10px_30px_var(--glow)]">
                    <ScanSearch className="h-4 w-4" /> Detect objects in this test image
                  </span>
                </span>
              </button>
            }
          />
        </ImageInput>

        <div className="mt-5">
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium">Held-out test images</p>
            <div className="ml-auto flex rounded-lg border border-line p-0.5 text-xs">
              {LAYOUTS.map((l) => (
                <button
                  key={l}
                  onClick={() => setLayout(l)}
                  className={`rounded-md px-2 py-1 capitalize ${layout === l ? "bg-surface-2 font-medium text-ink" : "text-ink-2"}`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-6 gap-2 sm:grid-cols-8">
            {composites.map((c) => (
              <button
                key={c.id}
                onClick={() => runSample(c)}
                className={`overflow-hidden rounded-md border transition ${activeSample === c.id ? "border-series-1 ring-2 ring-series-1" : "border-line hover:opacity-80"}`}
                aria-label={`Test image ${c.id} (${c.layout})`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={c.src} alt="" loading="lazy" className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <ModelStatus name="YOLOv8s detector" size="45 MB" state={model.state} />

        <div className="grid grid-cols-3 gap-2">
          <Stat label="Objects found" value={img ? String(shown.length) : "–"} />
          <Stat label="Inference" value={ms !== null ? `${ms.toFixed(0)} ms` : "–"} />
          <Stat label="Correct" value={scored && truth ? `${correct} / ${truth.length}` : "–"} hint={truth ? undefined : "test images only"} />
        </div>

        <label className="block text-sm">
          <span className="flex justify-between">
            <span className="font-medium">Confidence threshold</span>
            <span className="tabular text-ink-2">{conf.toFixed(2)}</span>
          </span>
          <input type="range" min={0.1} max={0.95} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} className="mt-2 w-full accent-[var(--series-1)]" />
        </label>
        {truth && (
          <label className="flex items-center gap-2 text-sm text-ink-2">
            <input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />
            Show ground-truth boxes (dashed white)
          </label>
        )}

        <div className="overflow-hidden rounded-xl border border-line bg-surface">
          <table className="w-full text-sm">
            <thead className="bg-surface-2 text-left text-xs text-ink-2">
              <tr>
                <th className="px-3 py-2 font-medium">Object ID</th>
                <th className="px-3 py-2 font-medium">Confidence</th>
                <th className="px-3 py-2 font-medium">Box (x1, y1, x2, y2)</th>
                {truth && <th className="px-3 py-2 font-medium">Status</th>}
              </tr>
            </thead>
            <tbody className="tabular">
              {shown.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-3 py-6 text-center text-ink-2">
                    {busy ? "Detecting…" : img ? "No objects above the threshold." : "Detections will appear here."}
                  </td>
                </tr>
              )}
              {[...shown]
                .sort((a, b) => a.y1 - b.y1 || a.x1 - b.x1)
                .map((d, i) => {
                  const st = scored?.status.get(d);
                  return (
                    <tr key={i} className="border-t border-line">
                      <td className="px-3 py-2">
                        <span className="flex items-center gap-2 font-medium">
                          {thumb(d.label) && (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={thumb(d.label)} alt="" className="h-7 w-7 rounded object-cover" />
                          )}
                          {d.label}
                        </span>
                      </td>
                      <td className="px-3 py-2">{(d.confidence * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2 text-ink-2">
                        {[d.x1, d.y1, d.x2, d.y2].map((v) => Math.round(v)).join(", ")}
                      </td>
                      {truth && (
                        <td className="px-3 py-2">
                          {st === "correct" ? (
                            <span className="inline-flex items-center gap-1 text-good-text">
                              <CheckCircle2 className="h-4 w-4" /> Correct
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-critical">
                              <CircleAlert className="h-4 w-4" /> Wrong
                            </span>
                          )}
                        </td>
                      )}
                    </tr>
                  );
                })}
            </tbody>
          </table>
          {scored && scored.missed.length > 0 && (
            <p className="border-t border-line px-3 py-2 text-xs text-critical">
              Missed: {scored.missed.map((m) => m.label).join(", ")}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface px-3 py-2.5">
      <p className="text-xs text-ink-2">{label}</p>
      <p className="tabular mt-0.5 text-lg font-semibold">{value}</p>
      {hint && <p className="text-[11px] text-muted">{hint}</p>}
    </div>
  );
}
