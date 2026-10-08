import type { Metadata } from "next";
import { m1, metrics, objectIds, samples } from "@/lib/data";

export const metadata: Metadata = { title: "Objects" };

export default function ObjectsPage() {
  const ap = Object.fromEntries(metrics.per_class.map((r) => [r.object_id, r.AP50]));
  return (
    <div className="mx-auto max-w-6xl px-4 pt-12 sm:px-6">
      <p className="text-sm font-medium text-brand">Dataset</p>
      <h1 className="mt-2 text-4xl font-semibold tracking-tight">All 73 objects</h1>
      <p className="mt-3 max-w-2xl text-ink-2">
        Held-out test photos of every object in the class dataset, with the detector&apos;s test AP@0.5 and the classifier&apos;s test F1 for that object.
      </p>
      <ul className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {objectIds.map((id) => (
          <li key={id} className="overflow-hidden rounded-xl border border-line bg-surface">
            <div className="grid grid-cols-3 gap-px bg-line">
              {samples.singles[id].map((src) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img key={src} src={src} alt={`${id} test photo`} loading="lazy" className="aspect-square w-full object-cover" />
              ))}
            </div>
            <div className="flex items-baseline justify-between px-3 py-2">
              <span className="font-semibold">{id}</span>
              <span className="tabular text-xs text-ink-2">
                {ap[id] !== undefined && `AP ${ap[id].toFixed(2)}`}
                {m1.per_class_f1[id] !== undefined && ` · F1 ${m1.per_class_f1[id].toFixed(2)}`}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
