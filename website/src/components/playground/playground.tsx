"use client";

import { Layers, ScanSearch, Tag } from "lucide-react";
import { useState } from "react";
import { DetectPanel } from "./detect-panel";
import { IdentifyPanel } from "./identify-panel";
import { ScenePanel } from "./scene-panel";

const TABS = [
  { id: "detect", label: "Detect", sub: "Multi-object · YOLOv8s", icon: ScanSearch },
  { id: "identify", label: "Identify", sub: "Single object · EfficientNet-B0", icon: Tag },
  { id: "scene", label: "Build a grid", sub: "Tile photos, then detect", icon: Layers },
] as const;

export function Playground() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("detect");
  const current = TABS.find((t) => t.id === tab)!;
  return (
    <div className="relative">
      <div className="pointer-events-none absolute inset-x-0 -top-10 bottom-0 bg-[radial-gradient(50%_40%_at_50%_0%,var(--glow),transparent_70%)]" />
      <section className="relative overflow-hidden rounded-[22px] border border-line-strong bg-surface shadow-[0_30px_80px_-20px_rgba(0,0,0,0.55)]">
        <div className="flex flex-wrap items-center gap-3 border-b border-line bg-surface-2/60 px-4 py-3">
          <div className="flex gap-1.5" aria-hidden>
            <span className="h-3 w-3 rounded-full bg-[#ff5f57]" />
            <span className="h-3 w-3 rounded-full bg-[#febc2e]" />
            <span className="h-3 w-3 rounded-full bg-[#28c840]" />
          </div>
          <div role="tablist" aria-label="Demo mode" className="mx-auto flex rounded-xl border border-line bg-page/60 p-1">
            {TABS.map((t) => {
              const Icon = t.icon;
              const on = tab === t.id;
              return (
                <button
                  key={t.id}
                  role="tab"
                  aria-selected={on}
                  onClick={() => setTab(t.id)}
                  className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm transition ${
                    on ? "bg-surface-3 text-ink shadow-sm" : "text-ink-2 hover:text-ink"
                  }`}
                >
                  <Icon className={`h-4 w-4 ${on ? "text-brand" : ""}`} />
                  {t.label}
                </button>
              );
            })}
          </div>
          <span className="hidden w-[180px] text-right text-xs text-muted md:block">{current.sub}</span>
        </div>
        <div className="p-4 sm:p-6">
          <div hidden={tab !== "detect"}><DetectPanel /></div>
          <div hidden={tab !== "identify"}><IdentifyPanel /></div>
          <div hidden={tab !== "scene"}><ScenePanel /></div>
        </div>
      </section>
    </div>
  );
}
