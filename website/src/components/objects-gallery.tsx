"use client";

import { AnimatePresence, motion } from "motion/react";
import dynamic from "next/dynamic";
import { useState } from "react";
import { m1, metrics, objectIds, photos } from "@/lib/data";
import { useCan3D } from "./three-utils";

const Objects3D = dynamic(() => import("./objects-3d"), { ssr: false });

const ITEMS = objectIds.map((id) => ({ id, src: photos[id][0] }));
const AP = Object.fromEntries(metrics.per_class.map((r) => [r.object_id, r]));

export function ObjectsGallery() {
  const can3D = useCan3D();
  const [selected, setSelected] = useState(objectIds[0]);
  if (!can3D) return null;
  const ap = AP[selected];
  const f1 = m1.per_class_f1[selected];

  return (
    <section className="relative mt-6 h-[640px] sm:h-[760px]" aria-label="All objects in 3D">
      <div className="absolute inset-y-0 left-1/2 w-screen -translate-x-1/2 [mask-image:linear-gradient(to_right,transparent,black_10%,black_90%,transparent)]">
        <Objects3D items={ITEMS} selected={selected} onSelect={setSelected} />
      </div>
      <p className="pointer-events-none absolute right-0 top-2 text-xs text-muted">Drag to spin · click an object</p>
      <div className="pointer-events-none absolute bottom-4 left-0 right-0 flex justify-center sm:justify-start">
        <AnimatePresence mode="wait">
          <motion.div
            key={selected}
            initial={{ opacity: 0, y: 16, filter: "blur(6px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -10, filter: "blur(6px)" }}
            transition={{ duration: 0.35 }}
            className="glass pointer-events-auto backdrop-blur-xl backdrop-saturate-150 w-full max-w-sm rounded-2xl border border-line-strong p-4 shadow-[0_20px_60px_-20px_rgba(0,0,0,0.6)]"
          >
            <div className="flex items-baseline justify-between">
              <p className="text-lg font-semibold tracking-tight">{selected}</p>
              <p className="tabular text-xs text-ink-2">
                {ap && `AP@0.5 ${ap.AP50.toFixed(2)}`}
                {f1 !== undefined && ` · F1 ${f1.toFixed(2)}`}
              </p>
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {photos[selected].map((src) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img key={src} src={src} alt={`${selected} test photo`} className="aspect-square w-full rounded-lg object-cover" />
              ))}
            </div>
            {ap && (
              <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
                <Meter label="Detector precision" value={ap.precision} />
                <Meter label="Detector recall" value={ap.recall} />
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex justify-between text-ink-2">
        <span>{label}</span>
        <span className="tabular text-ink">{value.toFixed(2)}</span>
      </div>
      <span className="mt-1 block h-1.5 overflow-hidden rounded-full bg-surface-3">
        <motion.span className="block h-full rounded-full bg-gradient-to-r from-brand to-brand-2" initial={{ width: 0 }} animate={{ width: `${value * 100}%` }} transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }} />
      </span>
    </div>
  );
}
