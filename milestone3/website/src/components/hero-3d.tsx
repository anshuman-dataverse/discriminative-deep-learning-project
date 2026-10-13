"use client";

import { Float, Line, Sparkles, useTexture } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { samples } from "@/lib/data";
import { type Label, parseLabels } from "@/lib/labels";
import { hue, tagTexture, useThemeColors } from "./three-utils";

const SIZE = 2.6;
const RADIUS = 6.2;
const STEP = 0.5;

function Boxes({ labels, active, names }: { labels: Label[]; active: boolean; names: string[] }) {
  const group = useRef<THREE.Group>(null);
  const start = useRef(0);
  const items = useMemo(
    () =>
      labels.map((l) => {
        const color = hue(l.c);
        const x1 = (l.x1 - 0.5) * SIZE, x2 = (l.x2 - 0.5) * SIZE;
        const y1 = (0.5 - l.y1) * SIZE, y2 = (0.5 - l.y2) * SIZE;
        return {
          color,
          points: [[x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0], [x1, y1, 0]] as [number, number, number][],
          tag: tagTexture(names[l.c] ?? `#${l.c}`, color),
          tagPos: [x1 + 0.27, y1 + 0.09, 0.002] as [number, number, number],
          center: new THREE.Vector3((x1 + x2) / 2, (y1 + y2) / 2, 0),
        };
      }),
    [labels, names],
  );

  useEffect(() => {
    start.current = performance.now();
  }, [active]);

  const stagger = Math.min(0.28, 2.2 / Math.max(1, labels.length));

  useFrame(() => {
    if (!group.current) return;
    const t = (performance.now() - start.current) / 1000;
    group.current.children.forEach((child, i) => {
      const k = active ? THREE.MathUtils.clamp((t - 0.35 - i * stagger) / 0.45, 0, 1) : 0;
      const e = 1 - Math.pow(1 - k, 3);
      child.visible = e > 0.01;
      child.scale.setScalar(0.85 + 0.15 * e);
      child.traverse((o) => {
        const m = (o as THREE.Mesh).material as THREE.Material & { opacity?: number };
        if (m && "opacity" in m) {
          m.transparent = true;
          m.opacity = e;
        }
      });
    });
  });

  return (
    <group ref={group} position={[0, 0, 0.012]}>
      {items.map((it, i) => (
        <group key={i} position={it.center}>
          <group position={[-it.center.x, -it.center.y, 0]}>
            <Line points={it.points} color={it.color} lineWidth={2.4} transparent />
            <mesh position={it.tagPos}>
              <planeGeometry args={[0.54, 0.135]} />
              <meshBasicMaterial map={it.tag} transparent toneMapped={false} />
            </mesh>
          </group>
        </group>
      ))}
    </group>
  );
}

function Panel({ src, labels, index, active, names }: { src: string; labels: Label[]; index: number; active: boolean; names: string[] }) {
  const texture = useTexture(src, (t) => {
    (Array.isArray(t) ? t : [t]).forEach((x) => (x.colorSpace = THREE.SRGBColorSpace));
  });
  const angle = (index - 3) * STEP;
  return (
    <group position={[Math.sin(angle) * RADIUS, 0, Math.cos(angle) * RADIUS]} rotation={[0, angle, 0]}>
      <Float speed={1.2} rotationIntensity={0.08} floatIntensity={0.25} floatingRange={[-0.06, 0.06]}>
        <mesh position={[0, 0, -0.02]}>
          <planeGeometry args={[SIZE + 0.08, SIZE + 0.08]} />
          <meshBasicMaterial color={active ? "#8ea2ff" : "#1d212d"} transparent opacity={active ? 0.55 : 0.9} toneMapped={false} />
        </mesh>
        <mesh>
          <planeGeometry args={[SIZE, SIZE]} />
          <meshBasicMaterial map={texture} toneMapped={false} />
        </mesh>
        <Boxes labels={labels} active={active} names={names} />
      </Float>
    </group>
  );
}

function CameraRig() {
  useFrame(({ camera, pointer }, dt) => {
    camera.position.x = THREE.MathUtils.damp(camera.position.x, pointer.x * 0.7, 2, dt);
    camera.position.y = THREE.MathUtils.damp(camera.position.y, 0.35 + pointer.y * 0.35, 2, dt);
    camera.lookAt(0, 0, -1.5);
  });
  return null;
}

function Carousel({ names }: { names: string[] }) {
  const picks = useMemo(() => {
    const byGrid = ["3x3", "2x2", "4x4", "3x3", "2x2", "4x4", "3x3"];
    const used = new Set<string>();
    return byGrid.map((g, i) => {
      const pool = samples.composites.filter((c) => c.grid === g && !used.has(c.id));
      const c = pool[i % Math.max(1, pool.length)] ?? samples.composites[i];
      used.add(c.id);
      return { src: c.src, labels: parseLabels(c.labels) };
    });
  }, []);
  const [active, setActive] = useState(3);
  const ring = useRef<THREE.Group>(null);

  useEffect(() => {
    const t = setInterval(() => setActive((a) => (a + 1) % picks.length), 3400);
    return () => clearInterval(t);
  }, [picks.length]);

  useFrame((_, dt) => {
    if (ring.current) ring.current.rotation.y = THREE.MathUtils.damp(ring.current.rotation.y, -(active - 3) * STEP, 2.4, dt);
  });

  return (
    <>
      <group position={[0, 0, -RADIUS]}>
        <group ref={ring}>
          {picks.map((p, i) => (
            <Panel key={p.src} src={p.src} labels={p.labels} index={i} active={i === active} names={names} />
          ))}
        </group>
      </group>
      <CameraRig />
    </>
  );
}

export default function Hero3D({ names, active = true }: { names: string[]; active?: boolean }) {
  const colors = useThemeColors();
  return (
    <Canvas
      frameloop={active ? "always" : "never"}
      dpr={[1, 1.75]}
      camera={{ position: [0, 0.35, 6.4], fov: 38 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      style={{ background: "transparent" }}
    >
      <fog attach="fog" args={[colors.page, 6.5, 15]} />
      <Sparkles count={70} scale={[12, 6, 6]} size={2.2} speed={0.25} opacity={0.5} color={colors.brand} />
      <Carousel names={names} />
    </Canvas>
  );
}
