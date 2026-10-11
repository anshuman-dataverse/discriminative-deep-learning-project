"use client";

import { Line, Sparkles, useTexture } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import type { MotionValue } from "motion/react";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { Label } from "@/lib/labels";
import { hue, tagTexture, useThemeColors } from "./three-utils";

const SIZE = 3.2;
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
const seg = (p: number, a: number, b: number) => clamp01((p - a) / (b - a));
const ease = (k: number) => 1 - Math.pow(1 - k, 3);
const easeInOut = (k: number) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);

const SCAN_A = 0.5;
const SCAN_B = 0.7;

type Card = {
  label: Label;
  color: THREE.Color;
  w: number;
  h: number;
  box: { x: number; y: number; w: number; h: number };
  home: THREE.Vector3;
  start: THREE.Vector3;
  spin: THREE.Euler;
  tex: THREE.Texture;
  tag: THREE.CanvasTexture;
  frame: [number, number, number][];
  scanAt: number;
};

function fade(obj: THREE.Object3D, opacity: number) {
  obj.visible = opacity > 0.005;
  obj.traverse((o) => {
    const m = (o as THREE.Mesh).material as (THREE.Material & { opacity: number }) | undefined;
    if (m && "opacity" in m) {
      m.transparent = true;
      m.opacity = opacity * ((m.userData.base as number | undefined) ?? 1);
    }
  });
}

const EXTRA_SPOTS: [number, number, number][] = [
  [-3.1, 1.5, -2.2], [2.9, 1.7, -2.6], [-2.4, -1.8, -1.6], [3.2, -1.4, -1.9],
  [0.4, 2.3, -3.4], [-0.6, -2.4, -3], [-4.2, 0.1, -3.6], [4.3, 0.3, -3.8],
];

function Extras({ progress, srcs }: { progress: MotionValue<number>; srcs: string[] }) {
  const textures = useTexture(srcs, (t) => {
    (Array.isArray(t) ? t : [t]).forEach((x) => (x.colorSpace = THREE.SRGBColorSpace));
  });
  const refs = useRef<(THREE.Group | null)[]>([]);
  useFrame(({ clock }) => {
    const p = progress.get();
    const out = ease(seg(p, 0.16, 0.34));
    refs.current.forEach((g, i) => {
      if (!g) return;
      const [x, y, z] = EXTRA_SPOTS[i];
      const t = clock.elapsedTime;
      g.position.set(x * (1 + out * 0.6), y * (1 + out * 0.6) + Math.sin(t * 0.7 + i) * 0.1, z - out * 2);
      g.rotation.set(Math.sin(t * 0.3 + i) * 0.12, -x * 0.12 + Math.cos(t * 0.25 + i) * 0.1, Math.sin(i * 1.9) * 0.15);
      fade(g, 1 - out);
    });
  });
  return (
    <>
      {textures.map((tex, i) => (
        <group key={i} ref={(el) => { refs.current[i] = el; }}>
          <mesh>
            <planeGeometry args={[0.95, 0.95]} />
            <meshBasicMaterial map={tex} transparent toneMapped={false} />
          </mesh>
        </group>
      ))}
    </>
  );
}

type GridShape = { cols: number; rows: number };

function Scene({ progress, composite, grid, extras, labels, names, colors }: {
  progress: MotionValue<number>;
  composite: string;
  grid: GridShape;
  extras: string[];
  labels: Label[];
  names: string[];
  colors: ReturnType<typeof useThemeColors>;
}) {
  const compTex = useTexture(composite, (t) => {
    (Array.isArray(t) ? t : [t]).forEach((x) => (x.colorSpace = THREE.SRGBColorSpace));
  });
  const viewport = useThree((s) => s.viewport);
  const narrow = viewport.aspect < 1;

  const cards = useMemo<Card[]>(() => {
    const n = labels.length;
    const { cols, rows } = grid;
    return labels.map((l, i) => {
      const col = Math.min(cols - 1, Math.floor(((l.x1 + l.x2) / 2) * cols));
      const row = Math.min(rows - 1, Math.floor(((l.y1 + l.y2) / 2) * rows));
      const tx = col / cols, ty = row / rows;
      const w = SIZE / cols;
      const h = SIZE / rows;
      const cx = (tx + 0.5 / cols - 0.5) * SIZE;
      const cy = (0.5 - ty - 0.5 / rows) * SIZE;
      const bw = (l.x2 - l.x1) * SIZE;
      const bh = (l.y2 - l.y1) * SIZE;
      const box = { x: ((l.x1 + l.x2) / 2 - tx - 0.5 / cols) * SIZE, y: (ty + 0.5 / rows - (l.y1 + l.y2) / 2) * SIZE, w: bw, h: bh };
      const tex = compTex.clone();
      tex.repeat.set(1 / cols, 1 / rows);
      tex.offset.set(tx, 1 - ty - 1 / rows);
      tex.needsUpdate = true;
      const a = (i / Math.max(1, n - 1) - 0.5) * Math.PI * 0.9;
      const color = hue(l.c);
      return {
        label: l,
        color,
        w,
        h,
        box,
        home: new THREE.Vector3(cx, cy, 0.01),
        start: new THREE.Vector3(Math.sin(a) * 2.5, (i % 2 ? 0.85 : -0.75) + Math.cos(i * 1.7) * 0.25, 1.3 - Math.abs(Math.sin(a)) * 1.2),
        spin: new THREE.Euler(0.25 * Math.cos(i * 2.1), -a * 0.55, 0.18 * Math.sin(i * 1.3)),
        tex,
        tag: tagTexture(names[l.c] ?? `#${l.c}`, color),
        frame: [[-bw / 2, bh / 2, 0], [bw / 2, bh / 2, 0], [bw / 2, -bh / 2, 0], [-bw / 2, -bh / 2, 0], [-bw / 2, bh / 2, 0]],
        scanAt: SCAN_A + (SCAN_B - SCAN_A) * clamp01(((l.y1 + l.y2) / 2 - 0.02)),
      };
    });
  }, [labels, names, compTex, grid]);

  const fly = Math.min(0.045, 0.2 / Math.max(1, cards.length));
  const tagStep = Math.min(0.025, 0.1 / Math.max(1, cards.length));

  const rig = useRef<THREE.Group>(null);
  const board = useRef<THREE.Group>(null);
  const bg = useRef<THREE.Mesh>(null);
  const rim = useRef<THREE.Mesh>(null);
  const scan = useRef<THREE.Group>(null);
  const cardRefs = useRef<(THREE.Group | null)[]>([]);
  const boxRefs = useRef<(THREE.Group | null)[]>([]);
  const tagRefs = useRef<(THREE.Mesh | null)[]>([]);
  const cur = useRef(0);
  const q = new THREE.Quaternion();
  const qs = new THREE.Quaternion();

  useFrame(({ camera, pointer, clock }, dt) => {
    cur.current = THREE.MathUtils.damp(cur.current, progress.get(), 6, dt);
    const p = cur.current;
    const t = clock.elapsedTime;

    const appear = seg(p, 0.12, 0.3);
    const explode = easeInOut(seg(p, 0.8, 0.97));
    const scanK = seg(p, SCAN_A, SCAN_B);

    if (bg.current) fade(bg.current, ease(appear));
    if (rim.current) fade(rim.current, ease(appear) * 0.7);

    cards.forEach((c, i) => {
      const g = cardRefs.current[i];
      if (!g) return;
      const k = easeInOut(seg(p, 0.2 + i * fly, 0.36 + i * fly));
      const bob = (1 - k) * 0.12;
      g.position.lerpVectors(c.start, c.home, k);
      g.position.y += Math.sin(t * 0.9 + i * 1.3) * bob;
      g.position.z += explode * (0.45 + i * 0.32);
      qs.setFromEuler(c.spin);
      q.identity();
      g.quaternion.slerpQuaternions(qs, q, k);
      const s = THREE.MathUtils.lerp(1.35, 1, k);
      g.scale.setScalar(s);

      const box = boxRefs.current[i];
      if (box) {
        const b = ease(seg(p, c.scanAt - 0.005, c.scanAt + 0.05));
        fade(box, b);
        box.scale.setScalar(1.18 - 0.18 * b);
      }
      const tag = tagRefs.current[i];
      if (tag) {
        const tk = ease(seg(p, 0.72 + i * tagStep, 0.78 + i * tagStep));
        fade(tag, tk);
        tag.position.y = c.box.y + c.box.h / 2 + 0.11 + (1 - tk) * 0.12;
      }
    });

    if (scan.current) {
      scan.current.position.y = THREE.MathUtils.lerp(SIZE / 2, -SIZE / 2, scanK);
      fade(scan.current, scanK > 0 && scanK < 1 ? Math.sin(scanK * Math.PI) : 0);
    }

    if (board.current) {
      board.current.rotation.y = THREE.MathUtils.lerp(0, narrow ? -0.45 : -0.62, explode);
      board.current.rotation.x = THREE.MathUtils.lerp(0, 0.22, explode);
    }

    if (rig.current) {
      rig.current.rotation.y = THREE.MathUtils.damp(rig.current.rotation.y, pointer.x * 0.12, 3, dt);
      rig.current.rotation.x = THREE.MathUtils.damp(rig.current.rotation.x, -pointer.y * 0.08, 3, dt);
    }

    const dist = (narrow ? 9.6 : 7.4) + (1 - appear) * 0.8 + explode * 0.6;
    camera.position.z = THREE.MathUtils.damp(camera.position.z, dist, 4, dt);
    camera.lookAt(0, 0, 0);
  });

  return (
    <group ref={rig}>
      <Extras progress={progress} srcs={extras} />
      <group ref={board}>
        <mesh ref={rim} position={[0, 0, -0.03]}>
          <planeGeometry args={[SIZE + 0.1, SIZE + 0.1]} />
          <meshBasicMaterial color={colors.brand} transparent opacity={0} toneMapped={false} />
        </mesh>
        <mesh ref={bg}>
          <planeGeometry args={[SIZE, SIZE]} />
          <meshBasicMaterial color={colors.surface} transparent opacity={0} toneMapped={false} />
        </mesh>

        {cards.map((c, i) => (
          <group key={i} ref={(el) => { cardRefs.current[i] = el; }}>
            <mesh position={[0, 0, -0.012]}>
              <planeGeometry args={[c.w + 0.05, c.h + 0.05]} />
              <meshBasicMaterial color="#000000" transparent opacity={0.35} toneMapped={false} />
            </mesh>
            <mesh>
              <planeGeometry args={[c.w, c.h]} />
              <meshBasicMaterial map={c.tex} toneMapped={false} />
            </mesh>
            <group ref={(el) => { boxRefs.current[i] = el; }} position={[c.box.x, c.box.y, 0.006]}>
              <Line points={c.frame} color={c.color} lineWidth={2.6} transparent />
              <mesh position={[0, 0, -0.001]}>
                <planeGeometry args={[c.box.w, c.box.h]} />
                <meshBasicMaterial color={c.color} transparent opacity={0} userData={{ base: 0.12 }} toneMapped={false} />
              </mesh>
            </group>
            <mesh ref={(el) => { tagRefs.current[i] = el; }} position={[c.box.x - c.box.w / 2 + 0.3, c.box.y + c.box.h / 2 + 0.11, 0.008]}>
              <planeGeometry args={[0.6, 0.15]} />
              <meshBasicMaterial map={c.tag} transparent opacity={0} toneMapped={false} />
            </mesh>
          </group>
        ))}

        <group ref={scan} position={[0, SIZE / 2, 0.05]}>
          <mesh>
            <planeGeometry args={[SIZE + 0.5, 0.025]} />
            <meshBasicMaterial color={colors.brand2} transparent toneMapped={false} />
          </mesh>
          <mesh position={[0, 0.22, -0.001]}>
            <planeGeometry args={[SIZE + 0.5, 0.44]} />
            <meshBasicMaterial transparent toneMapped={false} depthWrite={false}>
              <canvasTexture attach="map" args={[scanGradient()]} />
            </meshBasicMaterial>
          </mesh>
        </group>
      </group>
    </group>
  );
}

let gradientCanvas: HTMLCanvasElement | null = null;

function scanGradient() {
  if (gradientCanvas) return gradientCanvas;
  const c = document.createElement("canvas");
  c.width = 4;
  c.height = 128;
  const ctx = c.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 0, 128);
  g.addColorStop(0, "rgba(92,225,230,0)");
  g.addColorStop(1, "rgba(92,225,230,0.45)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 4, 128);
  gradientCanvas = c;
  return c;
}

export default function Story3D(props: {
  progress: MotionValue<number>;
  composite: string;
  grid: GridShape;
  extras: string[];
  labels: Label[];
  names: string[];
  active: boolean;
}) {
  const colors = useThemeColors();
  const { active, ...rest } = props;
  return (
    <Canvas
      frameloop={active ? "always" : "never"}
      dpr={[1, 1.75]}
      camera={{ position: [0, 0, 8.2], fov: 35 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      style={{ background: "transparent" }}
    >
      <fog attach="fog" args={[colors.page, 9, 18]} />
      <Sparkles count={60} scale={[10, 6, 4]} size={2} speed={0.2} opacity={0.45} color={colors.brand} />
      <Scene {...rest} colors={colors} />
    </Canvas>
  );
}
