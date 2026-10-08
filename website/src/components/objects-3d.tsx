"use client";

import { Sparkles, useCursor, useTexture } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { useThemeColors } from "./three-utils";

const ROWS = 3;
const CARD = 1;
const GAP = 1.22;

type Drag = { active: boolean; x: number; v: number; moved: number };

function Ring({ items, selected, onSelect, colors }: {
  items: { id: string; src: string }[];
  selected: string;
  onSelect: (id: string) => void;
  colors: ReturnType<typeof useThemeColors>;
}) {
  const drag = useRef<Drag>({ active: false, x: 0, v: 0.12, moved: 0 });
  const el = useThree((s) => s.gl.domElement);

  useEffect(() => {
    const d = drag.current;
    const down = (e: PointerEvent) => {
      d.active = true;
      d.x = e.clientX;
      d.moved = 0;
    };
    const move = (e: PointerEvent) => {
      if (!d.active) return;
      const dx = e.clientX - d.x;
      d.x = e.clientX;
      d.moved += Math.abs(dx);
      d.v = THREE.MathUtils.lerp(d.v, dx * 0.6, 0.5);
    };
    const up = () => {
      d.active = false;
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointerleave", up);
    el.addEventListener("pointercancel", up);
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointerleave", up);
      el.removeEventListener("pointercancel", up);
    };
  }, [el]);

  const textures = useTexture(items.map((i) => i.src), (t) => {
    (Array.isArray(t) ? t : [t]).forEach((x) => (x.colorSpace = THREE.SRGBColorSpace));
  });
  const perRow = Math.ceil(items.length / ROWS);
  const radius = (perRow * GAP) / (Math.PI * 2);
  const ring = useRef<THREE.Group>(null);
  const cards = useRef<(THREE.Group | null)[]>([]);
  const [hover, setHover] = useState<number | null>(null);
  useCursor(hover !== null);
  const viewport = useThree((s) => s.viewport);
  const narrow = viewport.aspect < 1;

  const layout = useMemo(
    () =>
      items.map((_, i) => {
        const row = i % ROWS;
        const col = Math.floor(i / ROWS);
        const inRow = Math.ceil((items.length - row) / ROWS);
        const angle = (col / inRow) * Math.PI * 2 + row * (Math.PI / perRow);
        return { angle, y: (1 - row) * (CARD + 0.22) };
      }),
    [items, perRow],
  );

  useFrame(({ camera }, dt) => {
    const d = drag.current;
    if (ring.current) {
      if (!d.active) d.v = THREE.MathUtils.damp(d.v, hover === null ? 0.12 : 0, 2, dt);
      ring.current.rotation.y += d.v * dt;
    }
    cards.current.forEach((g, i) => {
      if (!g) return;
      const on = hover === i || items[i].id === selected;
      const s = THREE.MathUtils.damp(g.scale.x, on ? 1.18 : 1, 8, dt);
      g.scale.setScalar(s);
      const r = radius + (on ? 0.35 : 0);
      const cur = Math.hypot(g.position.x, g.position.z);
      const next = THREE.MathUtils.damp(cur, r, 8, dt);
      const { angle } = layout[i];
      g.position.x = Math.sin(angle) * next;
      g.position.z = Math.cos(angle) * next;
    });
    camera.position.z = THREE.MathUtils.damp(camera.position.z, radius + (narrow ? 9 : 7.4), 4, dt);
    camera.lookAt(0, 0, 0);
  });

  return (
    <group ref={ring} position={[0, narrow ? 1.6 : 1.05, 0]} rotation={[0.08, 0, 0]}>
      {items.map((it, i) => {
        const { angle, y } = layout[i];
        const isSel = it.id === selected;
        return (
          <group
            key={it.id}
            ref={(el) => { cards.current[i] = el; }}
            position={[Math.sin(angle) * radius, y, Math.cos(angle) * radius]}
            rotation={[0, angle, 0]}
            onPointerOver={(e) => { e.stopPropagation(); setHover(i); }}
            onPointerOut={() => setHover((h) => (h === i ? null : h))}
            onClick={(e) => {
              e.stopPropagation();
              if (drag.current.moved < 6) onSelect(it.id);
            }}
          >
            <mesh position={[0, 0, -0.01]}>
              <planeGeometry args={[CARD + 0.06, CARD + 0.06]} />
              <meshBasicMaterial color={isSel ? colors.brand : colors.surface} toneMapped={false} />
            </mesh>
            <mesh>
              <planeGeometry args={[CARD, CARD]} />
              <meshBasicMaterial map={textures[i]} toneMapped={false} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

export default function Objects3D({ items, selected, onSelect }: { items: { id: string; src: string }[]; selected: string; onSelect: (id: string) => void }) {
  const colors = useThemeColors();
  return (
    <div className="h-full w-full cursor-grab active:cursor-grabbing">
      <Canvas dpr={[1, 1.75]} camera={{ position: [0, 0.4, 14], fov: 36 }} gl={{ antialias: true, alpha: true }} style={{ background: "transparent", touchAction: "pan-y" }}>
        <fog attach="fog" args={[colors.page, 10, 22]} />
        <Sparkles count={80} scale={[16, 8, 12]} size={2} speed={0.2} opacity={0.45} color={colors.brand} />
        <Ring items={items} selected={selected} onSelect={onSelect} colors={colors} />
      </Canvas>
    </div>
  );
}
