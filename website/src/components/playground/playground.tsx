"use client";

import { Layers, ScanSearch, Tag } from "lucide-react";
import { useState } from "react";
import { DetectPanel } from "./detect-panel";
import { IdentifyPanel } from "./identify-panel";
import { ScenePanel } from "./scene-panel";

const TABS = [
  { id: "detect", label: "Detect objects", sub: "Multi-object · YOLOv8s", icon: ScanSearch },
  { id: "identify", label: "Identify object", sub: "Single object · EfficientNet-B0", icon: Tag },
  { id: "scene", label: "Build a scene", sub: "Compose, then detect", icon: Layers },
] as const;

export function Playground() {
  const [tab, setTab] = useState<(typeof TABS)[number]["id"]>("detect");
  return (
    <section id="demo" className="rounded-2xl border border-line bg-page p-3 shadow-sm sm:p-5">
      <div role="tablist" className="mb-5 grid gap-2 sm:grid-cols-3">
        {TABS.map((t) => {
          const Icon = t.icon;
          const on = tab === t.id;
          return (
            <button
              key={t.id}
              role="tab"
              aria-selected={on}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-3 rounded-xl border px-4 py-3 text-left transition ${
                on ? "border-ink bg-surface shadow-sm" : "border-line bg-transparent hover:bg-surface"
              }`}
            >
              <span className={`grid h-9 w-9 place-items-center rounded-lg ${on ? "bg-brand text-white" : "bg-surface-2 text-ink-2"}`}>
                <Icon className="h-4 w-4" />
              </span>
              <span>
                <span className="block text-sm font-semibold">{t.label}</span>
                <span className="block text-xs text-ink-2">{t.sub}</span>
              </span>
            </button>
          );
        })}
      </div>
      <div hidden={tab !== "detect"}><DetectPanel /></div>
      <div hidden={tab !== "identify"}><IdentifyPanel /></div>
      <div hidden={tab !== "scene"}><ScenePanel /></div>
    </section>
  );
}
