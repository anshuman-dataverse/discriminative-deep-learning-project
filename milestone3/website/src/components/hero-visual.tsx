"use client";

import { useInView } from "motion/react";
import dynamic from "next/dynamic";
import { useRef } from "react";
import { classes, samples } from "@/lib/data";
import { useCan3D } from "./three-utils";

const Hero3D = dynamic(() => import("./hero-3d"), { ssr: false, loading: () => <Poster /> });

function Poster() {
  const picks = samples.composites.filter((c) => c.grid === "3x3").slice(0, 3);
  return (
    <div className="relative h-full w-full [perspective:1200px]">
      {picks.map((c, i) => (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={c.id}
          src={c.src}
          alt=""
          className="absolute left-1/2 top-1/2 w-[46%] rounded-xl border border-line-strong shadow-2xl"
          style={{ transform: `translate(-50%, -50%) translateX(${(i - 1) * 42}%) rotateY(${(1 - i) * 24}deg) scale(${i === 1 ? 1 : 0.86})`, zIndex: i === 1 ? 2 : 1, opacity: i === 1 ? 1 : 0.55 }}
        />
      ))}
    </div>
  );
}

export function HeroVisual() {
  const ok = useCan3D();
  const ref = useRef<HTMLDivElement>(null);
  const visible = useInView(ref, { margin: "100px 0px 100px 0px" });
  return (
    <div ref={ref} className="h-full w-full">
      {ok ? <Hero3D names={classes.detector} active={visible} /> : <Poster />}
    </div>
  );
}
