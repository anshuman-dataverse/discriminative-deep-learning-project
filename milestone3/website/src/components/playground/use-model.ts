"use client";

import { useCallback, useState } from "react";
import type { Backend } from "@/lib/ort";

export type ModelState =
  | { status: "idle" }
  | { status: "loading"; progress: number; startedAt?: number }
  | { status: "ready"; backend: Backend }
  | { status: "error"; message: string };

export function useModel(load: (onProgress: (f: number) => void) => Promise<{ backend: Backend }>) {
  const [state, setState] = useState<ModelState>({ status: "idle" });
  const ensure = useCallback(async () => {
    setState((s) => (s.status === "ready" ? s : { status: "loading", progress: 0 }));
    try {
      const { backend } = await load((p) =>
        setState((s) => ({
          status: "loading",
          progress: p,
          startedAt: p >= 1 ? (s.status === "loading" && s.startedAt) || Date.now() : undefined,
        })),
      );
      setState({ status: "ready", backend });
      return true;
    } catch (e) {
      setState({ status: "error", message: e instanceof Error ? e.message : String(e) });
      return false;
    }
  }, [load]);
  return { state, ensure };
}
