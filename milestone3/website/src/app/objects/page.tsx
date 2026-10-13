import type { Metadata } from "next";
import { Reveal } from "@/components/motion";
import { ObjectsGallery } from "@/components/objects-gallery";
import { PageHeader } from "@/components/ui";
import { classes, m1, metrics, objectIds, photos } from "@/lib/data";

export const metadata: Metadata = { title: "Objects" };

export default function ObjectsPage() {
  const excluded = classes.classifier.filter((id) => !classes.detector.includes(id));
  const ap = Object.fromEntries(metrics.per_class.map((r) => [r.object_id, r.AP50]));
  return (
    <div className="mx-auto max-w-6xl px-4 sm:px-6">
      <PageHeader kicker="Dataset" title={<>All <span className="font-serif font-normal italic text-gradient">{objectIds.length}</span> objects</>}>
        Held-out test photos of every object the detector knows, with its test AP@0.5 and the classifier&apos;s test F1 for that object.
        {excluded.length > 0 && ` ${excluded.join(", ")} was excluded from Milestone 2, so only the classifier knows it.`}
      </PageHeader>
      <ObjectsGallery />
      <ul className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {objectIds.map((id, i) => (
          <Reveal as="li" key={id} delay={(i % 4) * 0.05} y={20} className="overflow-hidden rounded-xl border border-line bg-surface transition-[transform,border-color,box-shadow] duration-300 hover:-translate-y-1 hover:border-line-strong hover:shadow-[0_18px_40px_-18px_var(--glow)]">
            <div className="grid grid-cols-3 gap-px bg-line">
              {photos[id].map((src) => (
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
          </Reveal>
        ))}
      </ul>
    </div>
  );
}
