"use client";

import { CheckCircle2, CircleAlert, Shuffle, Tag } from "lucide-react";
import { useCallback, useState } from "react";
import { classify, loadClassifier, type Prediction } from "@/lib/classifier";
import { classes, objectIds, samples, thumb } from "@/lib/data";
import { Stat } from "./detect-panel";
import { ImageInput, loadFromUrl, type LoadedImage } from "./image-input";
import { ImageStage } from "./image-stage";
import { ModelStatus } from "./model-status";
import { useModel } from "./use-model";

function pickSingles(n: number, random = true) {
  const ids = random
    ? [...objectIds].sort(() => Math.random() - 0.5).slice(0, n)
    : objectIds.filter((_, i) => i % 4 === 0).slice(0, n);
  return ids.map((id) => {
    const photos = samples.singles[id];
    return { id, src: random ? photos[Math.floor(Math.random() * photos.length)] : photos[0] };
  });
}

export function IdentifyPanel() {
  const model = useModel(loadClassifier);
  const [img, setImg] = useState<LoadedImage | null>(null);
  const [truth, setTruth] = useState<string | null>(null);
  const [top, setTop] = useState<Prediction[]>([]);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [gallery, setGallery] = useState(() => pickSingles(16, false));

  const run = useCallback(
    async (input: LoadedImage, gt: string | null) => {
      setImg(input);
      setTruth(gt);
      setTop([]);
      setBusy(true);
      if (await model.ensure()) {
        const r = await classify(input.image, classes.classifier, 5);
        setTop(r.top);
        setMs(r.ms);
      }
      setBusy(false);
    },
    [model],
  );

  const best = top[0];
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <div>
        <ImageInput onImage={(i) => run(i, null)}>
          <ImageStage
            src={img?.src ?? null}
            width={img?.width ?? 224}
            height={img?.height ?? 224}
            boxes={[]}
            busy={busy}
            empty={
              <div className="px-6 text-center text-sm text-ink-2">
                <Tag className="mx-auto mb-3 h-8 w-8 text-muted" />
                Pick a test photo below, or upload a photo of one class object.
              </div>
            }
          />
        </ImageInput>
        <div className="mt-5">
          <div className="mb-2 flex items-center">
            <p className="text-sm font-medium">Held-out single-object test photos</p>
            <button onClick={() => setGallery(pickSingles(16))} className="ml-auto inline-flex items-center gap-1 text-xs text-ink-2 hover:text-ink">
              <Shuffle className="h-3.5 w-3.5" /> Shuffle
            </button>
          </div>
          <div className="grid grid-cols-8 gap-2">
            {gallery.map((g) => (
              <button key={g.src} onClick={async () => run(await loadFromUrl(g.src), g.id)} className="overflow-hidden rounded-md border border-line hover:opacity-80" aria-label={`Test photo of ${g.id}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={g.src} alt="" loading="lazy" className="aspect-square w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <ModelStatus name="EfficientNet-B0 classifier" size="16 MB" state={model.state} />
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Prediction" value={best ? best.label : "–"} />
          <Stat label="Confidence" value={best ? `${(best.probability * 100).toFixed(1)}%` : "–"} />
          <Stat label="Inference" value={ms !== null && best ? `${ms.toFixed(0)} ms` : "–"} />
        </div>

        {best && (
          <div className="flex items-center gap-4 rounded-xl border border-line bg-surface p-4">
            {thumb(best.label) && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={thumb(best.label)} alt={`Reference photo of ${best.label}`} className="h-20 w-20 rounded-lg object-cover" />
            )}
            <div>
              <p className="text-xs text-ink-2">Identified as</p>
              <p className="text-2xl font-semibold tracking-tight">{best.label}</p>
              {truth && (
                <p className={`mt-1 inline-flex items-center gap-1 text-sm ${truth === best.label ? "text-good-text" : "text-critical"}`}>
                  {truth === best.label ? <CheckCircle2 className="h-4 w-4" /> : <CircleAlert className="h-4 w-4" />}
                  {truth === best.label ? "Correct" : `Wrong (true: ${truth})`}
                </p>
              )}
            </div>
          </div>
        )}

        <div className="rounded-xl border border-line bg-surface p-4">
          <p className="mb-3 text-sm font-medium">Top-5 Object IDs</p>
          {top.length === 0 ? (
            <p className="py-4 text-center text-sm text-ink-2">{busy ? "Identifying…" : "Probabilities will appear here."}</p>
          ) : (
            <ul className="space-y-2.5">
              {top.map((p) => (
                <li key={p.label} className="grid grid-cols-[4.5rem_1fr_3.5rem] items-center gap-3 text-sm">
                  <span className="font-medium">{p.label}</span>
                  <span className="h-2 overflow-hidden rounded-full bg-surface-2">
                    <span className="block h-full rounded-full bg-series-1" style={{ width: `${Math.max(1, p.probability * 100)}%` }} />
                  </span>
                  <span className="tabular text-right text-ink-2">{(p.probability * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <p className="text-xs text-muted">
          The classifier answers “which one of the 73 objects is this?”, so it always picks one. For photos with several objects, use Detect.
        </p>
      </div>
    </div>
  );
}
