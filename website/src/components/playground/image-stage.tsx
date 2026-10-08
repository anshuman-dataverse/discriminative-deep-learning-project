"use client";

import { classColor } from "@/lib/colors";

export type StageBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  classIndex: number;
  confidence?: number;
  status?: "correct" | "wrong";
  truth?: boolean;
};

/** An image with boxes drawn over it in the image's own pixel coordinates. */
export function ImageStage({
  src,
  width,
  height,
  boxes,
  busy,
  empty,
}: {
  src: string | null;
  width: number;
  height: number;
  boxes: StageBox[];
  busy?: boolean;
  empty?: React.ReactNode;
}) {
  const stroke = Math.max(2, Math.round(Math.max(width, height) / 220));
  const font = Math.max(12, Math.round(Math.max(width, height) / 38));
  return (
    <div className="relative grid aspect-square w-full place-items-center overflow-hidden rounded-xl border border-line bg-surface dot-grid">
      {src ? (
        <div className="relative max-h-full max-w-full" style={{ aspectRatio: `${width} / ${height}`, width: width >= height ? "100%" : "auto", height: height > width ? "100%" : "auto" }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt="Input" className="h-full w-full object-contain" draggable={false} />
          <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" aria-hidden>
            {boxes.map((b, i) => {
              const color = b.truth ? "#ffffff" : b.status === "wrong" ? "#d03b3b" : b.status === "correct" ? "#0ca30c" : classColor(b.classIndex);
              const text = b.confidence !== undefined ? `${b.label} ${(b.confidence * 100).toFixed(0)}%` : b.label;
              const tw = text.length * font * 0.58 + 10;
              const ty = b.y1 - font - 6 >= 0 ? b.y1 - font - 6 : b.y1;
              return (
                <g key={i}>
                  <rect
                    x={b.x1}
                    y={b.y1}
                    width={b.x2 - b.x1}
                    height={b.y2 - b.y1}
                    fill="none"
                    stroke={color}
                    strokeWidth={stroke}
                    strokeDasharray={b.truth ? `${stroke * 3} ${stroke * 2}` : undefined}
                    rx={stroke}
                  />
                  {!b.truth && (
                    <>
                      <rect x={b.x1} y={ty} width={tw} height={font + 6} fill={color} rx={stroke} />
                      <text x={b.x1 + 5} y={ty + font} fontSize={font} fontWeight={600} fill="#0b0b0b" fontFamily="ui-sans-serif, system-ui">
                        {text}
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      ) : (
        empty
      )}
      {busy && (
        <div className="absolute inset-0 grid place-items-center bg-page/40 backdrop-blur-[1px]">
          <span className="rounded-full bg-surface px-3 py-1 text-xs font-medium shadow">Running model…</span>
        </div>
      )}
    </div>
  );
}
