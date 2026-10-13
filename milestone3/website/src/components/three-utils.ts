"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import * as THREE from "three";

export function hue(c: number) {
  return new THREE.Color().setHSL((c * 0.618034) % 1, 0.85, 0.6);
}

export function tagTexture(text: string, color: THREE.Color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = `#${color.getHexString()}`;
  ctx.beginPath();
  ctx.roundRect(0, 6, 240, 52, 12);
  ctx.fill();
  ctx.fillStyle = "#07080b";
  ctx.font = "600 34px ui-sans-serif, system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 16, 33);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export function useThemeColors() {
  const read = () => {
    const css = getComputedStyle(document.documentElement);
    return {
      page: css.getPropertyValue("--page").trim() || "#07080b",
      brand: css.getPropertyValue("--brand").trim() || "#8ea2ff",
      brand2: css.getPropertyValue("--brand-2").trim() || "#5ce1e6",
      surface: css.getPropertyValue("--surface-3").trim() || "#1d212d",
    };
  };
  const [colors, setColors] = useState(read);
  useEffect(() => {
    const obs = new MutationObserver(() => setColors(read()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);
  return colors;
}

let canUse3D: boolean | null = null;

function check3D() {
  if (canUse3D === null) {
    let gl = false;
    try {
      gl = !!document.createElement("canvas").getContext("webgl2");
    } catch {}
    canUse3D = gl && !matchMedia("(prefers-reduced-motion: reduce)").matches;
  }
  return canUse3D;
}

const subscribe = () => () => {};

export function useCan3D() {
  return useSyncExternalStore(subscribe, check3D, () => false);
}
