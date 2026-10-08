import type { ModelState } from "./use-model";

export function ModelStatus({ name, size, state }: { name: string; size: string; state: ModelState }) {
  let text: string;
  let dot = "bg-muted";
  if (state.status === "idle") text = `${name} loads on first use (${size})`;
  else if (state.status === "loading") {
    text = `Downloading ${name}… ${Math.round(state.progress * 100)}%`;
    dot = "bg-series-1 animate-pulse";
  } else if (state.status === "ready") {
    text = `${name} ready · ${state.backend === "webgpu" ? "WebGPU (GPU)" : "WebAssembly (CPU)"}`;
    dot = "bg-good";
  } else {
    text = `${name} failed to load: ${state.message}`;
    dot = "bg-critical";
  }
  return (
    <div className="flex items-center gap-2 text-xs text-ink-2" role="status">
      <span className={`h-2 w-2 shrink-0 rounded-full ${dot}`} />
      <span>{text}</span>
      {state.status === "loading" && (
        <span className="ml-1 h-1 w-24 overflow-hidden rounded-full bg-surface-2">
          <span className="block h-full bg-series-1 transition-[width]" style={{ width: `${state.progress * 100}%` }} />
        </span>
      )}
    </div>
  );
}
