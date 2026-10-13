"use client";

import { Dices, Wand2 } from "lucide-react";
import { useMemo, useState } from "react";
import { composeGrid } from "@/lib/compose";
import { classes, GRIDS, gridShape, objectIds, samples, thumb, TILE, type Grid } from "@/lib/data";
import { detect, loadDetector, type Detection } from "@/lib/detector";
import { score, Stat } from "./detect-panel";
import { ImageStage, type StageBox } from "./image-stage";
import { ModelStatus } from "./model-status";
import { useModel } from "./use-model";

const pick = <T,>(xs: T[]) => xs[Math.floor(Math.random() * xs.length)];
const shuffle = <T,>(xs: T[]) => [...xs].sort(() => Math.random() - 0.5);
const cellsOf = (g: Grid) => gridShape(g).cols * gridShape(g).rows;

export function ScenePanel() {
  const model = useModel(loadDetector);
  const [chosen, setChosen] = useState<string[]>(["OBJ011", "OBJ020", "OBJ027", "OBJ053"]);
  const [grid, setGrid] = useState<Grid>("3x3");
  const [scene, setScene] = useState<{ src: string; width: number; height: number; truth: StageBox[] } | null>(null);
  const [dets, setDets] = useState<Detection[]>([]);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState("");

  const cells = cellsOf(grid);
  const toggle = (id: string) =>
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : c.length < cells ? [...c, id] : c));

  const build = async (ids = chosen, g = grid) => {
    const n = cellsOf(g);
    const own = ids.slice(0, n);
    const fill = shuffle(objectIds.filter((id) => !own.includes(id))).slice(0, n - own.length);
    setBusy(true);
    setDets([]);
    const { canvas, placed } = await composeGrid(
      g,
      shuffle([...own, ...fill]).map((id) => ({ ...pick(samples.singles[id]), label: id })),
    );
    const truth = placed.map((p) => ({ ...p, classIndex: classes.detector.indexOf(p.label), truth: true }));
    setScene({ src: canvas.toDataURL("image/jpeg", 0.92), width: canvas.width, height: canvas.height, truth });
    if (await model.ensure()) {
      const r = await detect(canvas, canvas.width, canvas.height, classes.detector, 0.5);
      setDets(r.detections);
      setMs(r.ms);
    }
    setBusy(false);
  };

  const scored = useMemo(() => (scene ? score(dets, scene.truth) : null), [dets, scene]);
  const correct = scored ? [...scored.status.values()].filter((s) => s === "correct").length : 0;
  const boxes: StageBox[] = scene ? [...scene.truth, ...dets.map((d) => ({ ...d, status: scored?.status.get(d) }))] : [];
  const visible = objectIds.filter((id) => id.toLowerCase().includes(filter.trim().toLowerCase()));

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <div>
        <ImageStage
          src={scene?.src ?? null}
          width={scene?.width ?? TILE * gridShape(grid).cols}
          height={scene?.height ?? TILE * gridShape(grid).rows}
          boxes={boxes}
          busy={busy}
          empty={
            <div className="px-6 text-center text-sm text-ink-2">
              <Wand2 className="mx-auto mb-3 h-8 w-8 text-muted" />
              Choose a grid size and up to {cells} objects; empty cells get random objects. The grid of 224×224 test photos is built in your browser exactly like our training data, then the detector runs on it.
            </div>
          }
        />
        {scored && (
          <p className="mt-3 text-sm text-ink-2">
            Dashed white = the labelled object box in each tile (ground truth). Green = correct ID and location, red = wrong.
            {scored.missed.length > 0 && <span className="text-critical"> Missed: {scored.missed.map((m) => m.label).join(", ")}.</span>}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-4">
        <ModelStatus name="YOLOv8s detector" size="45 MB" state={model.state} />
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Objects placed" value={scene ? String(scene.truth.length) : "–"} />
          <Stat label="Correct" value={scene && !busy ? `${correct} / ${scene.truth.length}` : "–"} />
          <Stat label="Inference" value={ms !== null && !busy ? `${ms.toFixed(0)} ms` : "–"} />
        </div>

        <div>
          <p className="mb-2 text-sm font-medium">Grid size</p>
          <div className="grid grid-cols-5 gap-2">
            {GRIDS.map((g) => (
              <button
                key={g}
                onClick={() => {
                  setGrid(g);
                  setChosen((c) => c.slice(0, cellsOf(g)));
                }}
                aria-pressed={grid === g}
                className={`rounded-lg border px-2 py-2 text-left text-sm ${grid === g ? "border-series-1 bg-surface ring-1 ring-series-1" : "border-line bg-surface hover:bg-surface-2"}`}
              >
                <span className="block font-medium tabular">{g.replace("x", "×")}</span>
                <span className="block text-[11px] leading-tight text-ink-2">{cellsOf(g)} objects</span>
              </button>
            ))}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2">
            <p className="text-sm font-medium">Objects <span className="tabular text-ink-2">({chosen.length}/{cells})</span></p>
            <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter, e.g. 020" className="ml-auto w-32 rounded-md border border-line bg-surface px-2 py-1 text-xs" />
          </div>
          <div className="grid max-h-56 grid-cols-6 gap-1.5 overflow-y-auto rounded-lg border border-line bg-surface p-2 sm:grid-cols-8">
            {visible.map((id) => {
              const on = chosen.includes(id);
              return (
                <button key={id} onClick={() => toggle(id)} title={id} aria-pressed={on} className={`relative overflow-hidden rounded-md border ${on ? "border-series-1 ring-2 ring-series-1" : "border-line opacity-80 hover:opacity-100"}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={thumb(id)} alt={id} loading="lazy" className="aspect-square w-full object-cover" />
                  <span className="absolute inset-x-0 bottom-0 bg-black/55 text-center text-[9px] font-medium text-white">{id.slice(3)}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex gap-2">
          <button onClick={() => build()} disabled={busy} className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-ink px-3 py-2.5 text-sm font-medium text-page hover:opacity-90 disabled:opacity-50">
            <Wand2 className="h-4 w-4" /> Build grid &amp; detect
          </button>
          <button
            onClick={() => {
              const g = pick(GRIDS);
              setGrid(g);
              setChosen([]);
              build([], g);
            }}
            disabled={busy}
            className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2.5 text-sm font-medium hover:bg-surface-2 disabled:opacity-50"
          >
            <Dices className="h-4 w-4" /> Random
          </button>
        </div>
      </div>
    </div>
  );
}
