"use client";

import { animate, motion, useInView, useMotionTemplate, useMotionValue, useReducedMotion, useScroll, useSpring, useTransform } from "motion/react";
import { useEffect, useRef, useState } from "react";

const EASE = [0.2, 0.7, 0.2, 1] as const;

export function Reveal({
  children,
  className,
  delay = 0,
  y = 28,
  as = "div",
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
  y?: number;
  as?: "div" | "li" | "section" | "figure";
}) {
  const reduce = useReducedMotion();
  const Tag = motion[as];
  return (
    <Tag
      className={className}
      initial={reduce ? false : { opacity: 0, y, filter: "blur(6px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "0px 0px -12% 0px" }}
      transition={{ duration: 0.9, ease: EASE, delay }}
    >
      {children}
    </Tag>
  );
}

const NUM = /-?\d[\d,]*(?:\.\d+)?/;

export function CountUp({ value, className, duration = 1.6 }: { value: string; className?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -10% 0px" });
  const reduce = useReducedMotion();
  const match = value.match(NUM);
  const [text, setText] = useState(value);

  useEffect(() => {
    if (!match || reduce || !inView) return;
    const raw = match[0];
    const target = Number(raw.replace(/,/g, ""));
    const decimals = raw.includes(".") ? raw.split(".")[1].length : 0;
    const grouped = raw.includes(",");
    const fmt = (n: number) => {
      const s = n.toFixed(decimals);
      return grouped ? Number(s).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : s;
    };
    const controls = animate(0, target, {
      duration,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (n) => setText(value.replace(raw, fmt(n))),
    });
    return () => controls.stop();
  }, [inView, reduce]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <span ref={ref} className={className}>
      {text}
    </span>
  );
}

export function Tilt({ children, className, max = 8 }: { children: React.ReactNode; className?: string; max?: number }) {
  const reduce = useReducedMotion();
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), { stiffness: 160, damping: 18 });
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), { stiffness: 160, damping: 18 });
  const gx = useTransform(px, (v) => `${v * 100}%`);
  const gy = useTransform(py, (v) => `${v * 100}%`);
  const glare = useMotionTemplate`radial-gradient(420px circle at ${gx} ${gy}, var(--glow), transparent 60%)`;

  return (
    <motion.div
      className={`relative [transform-style:preserve-3d] ${className ?? ""}`}
      style={reduce ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: 900 }}
      onPointerMove={(e) => {
        if (e.pointerType !== "mouse") return;
        const r = e.currentTarget.getBoundingClientRect();
        px.set((e.clientX - r.left) / r.width);
        py.set((e.clientY - r.top) / r.height);
      }}
      onPointerLeave={() => {
        px.set(0.5);
        py.set(0.5);
      }}
    >
      {children}
      <motion.span aria-hidden className="pointer-events-none absolute inset-0 rounded-[inherit] opacity-0 transition-opacity duration-300 [.group:hover_&]:opacity-100" style={{ background: glare }} />
    </motion.div>
  );
}

export function Bar({ value, className, delay = 0 }: { value: number; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.span
      className={`block h-full origin-left rounded-full ${className ?? ""}`}
      style={{ width: `${value * 100}%` }}
      initial={reduce ? false : { scaleX: 0 }}
      whileInView={{ scaleX: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1], delay }}
    />
  );
}

export function HeroScroll({ text, visual, className }: { text: React.ReactNode; visual: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const textY = useTransform(scrollYProgress, [0, 1], [0, -90]);
  const textO = useTransform(scrollYProgress, [0, 0.7], [1, 0]);
  const visY = useTransform(scrollYProgress, [0, 1], [0, 140]);
  const visS = useTransform(scrollYProgress, [0, 1], [1, 0.9]);
  const visO = useTransform(scrollYProgress, [0.3, 0.95], [1, 0]);
  return (
    <div ref={ref} className={className}>
      <motion.div className="relative z-10" style={reduce ? undefined : { y: textY, opacity: textO }}>
        {text}
      </motion.div>
      <motion.div className="relative" style={reduce ? undefined : { y: visY, scale: visS, opacity: visO }}>
        {visual}
      </motion.div>
    </div>
  );
}

export function ScrollProgress() {
  const reduce = useReducedMotion();
  const p = useMotionValue(0);
  const scaleX = useSpring(p, { stiffness: 200, damping: 30 });
  useEffect(() => {
    const on = () => {
      const h = document.documentElement.scrollHeight - innerHeight;
      p.set(h > 0 ? scrollY / h : 0);
    };
    on();
    addEventListener("scroll", on, { passive: true });
    addEventListener("resize", on);
    return () => {
      removeEventListener("scroll", on);
      removeEventListener("resize", on);
    };
  }, [p]);
  if (reduce) return null;
  return <motion.div aria-hidden className="fixed inset-x-0 top-0 z-50 h-[2px] origin-left bg-gradient-to-r from-brand to-brand-2" style={{ scaleX }} />;
}
