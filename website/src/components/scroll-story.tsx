"use client";

import { AnimatePresence, motion, useInView, useMotionValueEvent, useScroll } from "motion/react";
import dynamic from "next/dynamic";
import { useRef, useState } from "react";
import { classes, samples } from "@/lib/data";
import { type Label, parseLabels } from "@/lib/labels";
import { classColor } from "@/lib/colors";
import { useCan3D } from "./three-utils";

const Story3D = dynamic(() => import("./story-3d"), { ssr: false });

export type Step = { kicker: string; title: string; body: string; stat: string; statLabel: string };

function overlaps(a: Label, b: Label) {
  return a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;
}

function pickScene() {
  const ok = samples.composites.find((c) => {
    const ls = parseLabels(c.labels);
    return c.layout === "scatter" && ls.length >= 4 && ls.every((a, i) => ls.every((b, j) => i === j || !overlaps(a, b)));
  });
  const c = ok ?? samples.composites[0];
  const labels = parseLabels(c.labels);
  const used = new Set(labels.map((l) => classes.detector[l.c]));
  const extras = Object.keys(samples.singles).filter((id) => !used.has(id)).filter((_, i) => i % 9 === 4).slice(0, 8).map((id) => samples.singles[id][0]);
  return { src: c.src, labels, background: samples.backgrounds[0], extras };
}

const SCENE = pickScene();

export function ScrollStory({ steps }: { steps: Step[] }) {
  const can3D = useCan3D();
  if (!can3D) return <StaticStory steps={steps} scene={SCENE} />;
  return <PinnedStory steps={steps} scene={SCENE} />;
}

function PinnedStory({ steps, scene }: { steps: Step[]; scene: typeof SCENE }) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const near = useInView(ref, { margin: "200px 0px 200px 0px" });
  const [step, setStep] = useState(0);
  const [p, setP] = useState(0);

  useMotionValueEvent(scrollYProgress, "change", (v) => {
    setP(v);
    setStep(Math.min(steps.length - 1, Math.floor(v * steps.length * 0.999)));
  });

  return (
    <section ref={ref} className="relative h-[460vh]" aria-label="How a detection is made">
      <div className="sticky top-0 flex h-svh flex-col overflow-hidden lg:grid lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)] lg:items-center">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_65%_50%,var(--glow),transparent_70%)]" />
        <div className="relative order-2 mx-auto w-full max-w-6xl px-4 pb-8 sm:px-6 lg:order-1 lg:mx-0 lg:max-w-none lg:pb-0 lg:pl-[max(1.5rem,calc((100vw-72rem)/2+1.5rem))]">
          <p className="text-sm font-medium text-brand">How it works</p>
          <div className="relative mt-3 min-h-[13.5rem] sm:min-h-[12rem]">
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                initial={{ opacity: 0, y: 24, filter: "blur(8px)" }}
                animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                exit={{ opacity: 0, y: -16, filter: "blur(8px)" }}
                transition={{ duration: 0.45, ease: [0.2, 0.7, 0.2, 1] }}
              >
                <p className="tabular text-xs font-medium uppercase tracking-[0.18em] text-muted">
                  {String(step + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")} · {steps[step].kicker}
                </p>
                <h2 className="mt-2 text-3xl font-semibold tracking-[-0.03em] sm:text-5xl">{steps[step].title}</h2>
                <p className="mt-3 max-w-md text-base leading-relaxed text-ink-2 sm:text-lg">{steps[step].body}</p>
                <p className="mt-5 flex items-baseline gap-2">
                  <span className="tabular text-gradient text-3xl font-semibold tracking-tight">{steps[step].stat}</span>
                  <span className="text-sm text-muted">{steps[step].statLabel}</span>
                </p>
              </motion.div>
            </AnimatePresence>
          </div>
          <ol className="mt-6 flex gap-2" aria-hidden>
            {steps.map((s, i) => {
              const k = Math.min(1, Math.max(0, p * steps.length - i));
              return (
                <li key={s.kicker} className="flex-1">
                  <span className="block h-1 overflow-hidden rounded-full bg-surface-3">
                    <span className="block h-full origin-left rounded-full bg-gradient-to-r from-brand to-brand-2" style={{ transform: `scaleX(${k})` }} />
                  </span>
                  <span className={`mt-2 block text-[11px] transition-colors ${i === step ? "text-ink" : "text-muted"}`}>{s.kicker}</span>
                </li>
              );
            })}
          </ol>
        </div>
        <div className="relative order-1 min-h-0 flex-1 lg:order-2 lg:h-full lg:[mask-image:linear-gradient(to_right,transparent,black_14%)]">
          <Story3D progress={scrollYProgress} composite={scene.src} background={scene.background} extras={scene.extras} labels={scene.labels} names={classes.detector} active={near} />
          <p className="absolute inset-x-0 bottom-3 text-center text-[11px] text-muted lg:bottom-8">Built from a real test image and its label file</p>
        </div>
      </div>
    </section>
  );
}

function StaticStory({ steps, scene }: { steps: Step[]; scene: typeof SCENE }) {
  return (
    <section className="mx-auto max-w-6xl px-4 pt-24 sm:px-6" aria-label="How a detection is made">
      <p className="text-sm font-medium text-brand">How it works</p>
      <div className="mt-6 grid items-center gap-8 lg:grid-cols-2">
        <ol className="space-y-6">
          {steps.map((s, i) => (
            <li key={s.kicker}>
              <p className="tabular text-xs font-medium uppercase tracking-[0.18em] text-muted">{String(i + 1).padStart(2, "0")} · {s.kicker}</p>
              <h3 className="mt-1 text-xl font-semibold tracking-tight">{s.title}</h3>
              <p className="mt-1 text-ink-2">{s.body}</p>
            </li>
          ))}
        </ol>
        <div className="relative overflow-hidden rounded-2xl border border-line">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={scene.src} alt="Generated multi-object test image" className="w-full" />
          {scene.labels.map((l, i) => (
            <span
              key={i}
              className="absolute rounded-sm border-2"
              style={{ left: `${l.x1 * 100}%`, top: `${l.y1 * 100}%`, width: `${(l.x2 - l.x1) * 100}%`, height: `${(l.y2 - l.y1) * 100}%`, borderColor: classColor(l.c) }}
            >
              <span className="absolute -top-5 left-0 rounded px-1 text-[10px] font-semibold text-black" style={{ background: classColor(l.c) }}>
                {classes.detector[l.c]}
              </span>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
