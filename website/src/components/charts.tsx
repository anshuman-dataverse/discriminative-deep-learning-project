"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type Format = "fixed3" | "fixed2" | "pct";
const FORMATS: Record<Format, (v: number) => string> = {
  fixed3: (v) => v.toFixed(3),
  fixed2: (v) => v.toFixed(2),
  pct: (v) => `${(v * 100).toFixed(1)}%`,
};

const axis = { stroke: "var(--axis)", tick: { fill: "var(--muted)", fontSize: 11 }, tickLine: false };
const grid = <CartesianGrid stroke="var(--grid)" vertical={false} />;

function TooltipBox({ title, rows }: { title: string; rows: { label: string; value: string; color?: string }[] }) {
  return (
    <div className="rounded-lg border border-line bg-surface px-3 py-2 text-xs shadow-md">
      <p className="mb-1 font-semibold">{title}</p>
      {rows.map((r) => (
        <p key={r.label} className="flex items-center gap-2 text-ink-2">
          {r.color && <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />}
          <span>{r.label}</span>
          <span className="tabular ml-auto pl-3 font-medium text-ink">{r.value}</span>
        </p>
      ))}
    </div>
  );
}

export type Series = { key: string; label: string; color: string };

export function Legend({ series }: { series: Series[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-4 text-xs text-ink-2">
      {series.map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1.5">
          <span className="h-0.5 w-4 rounded" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
    </div>
  );
}

/** Lines over epochs on one shared axis. */
export function EpochLines({
  data,
  series,
  domain,
  format: fmt = "fixed3",
}: {
  data: Record<string, number>[];
  series: Series[];
  domain?: [number, number];
  format?: Format;
}) {
  const format = FORMATS[fmt];
  return (
    <div>
      {series.length > 1 && <Legend series={series} />}
      <div className="h-64">
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
            {grid}
            <XAxis dataKey="epoch" {...axis} />
            <YAxis {...axis} domain={domain ?? ["auto", "auto"]} tickFormatter={(v) => format(v)} width={52} />
            <Tooltip
              cursor={{ stroke: "var(--axis)" }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <TooltipBox
                    title={`Epoch ${label}`}
                    rows={series.map((s) => ({
                      label: s.label,
                      color: s.color,
                      value: format(Number(payload.find((p) => p.dataKey === s.key)?.value ?? NaN)),
                    }))}
                  />
                ) : null
              }
            />
            {series.map((s) => (
              <Line key={s.key} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** One bar per category, single series, zero baseline. */
export function Bars({
  data,
  xKey,
  yKey,
  label,
  format: fmt = "fixed3",
  height = 256,
  highlightBelow,
  domain = [0, 1],
  showTicks = true,
  gap = 2,
}: {
  data: Record<string, string | number>[];
  xKey: string;
  yKey: string;
  label: string;
  format?: Format;
  height?: number;
  highlightBelow?: number;
  domain?: [number, number];
  showTicks?: boolean;
  gap?: number | string;
}) {
  const format = FORMATS[fmt];
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -8 }} barCategoryGap={gap}>
          {grid}
          <XAxis dataKey={xKey} {...axis} interval={showTicks ? 0 : "preserveStartEnd"} tick={showTicks ? axis.tick : false} />
          <YAxis {...axis} domain={domain} tickFormatter={(v) => format(v)} width={52} />
          <Tooltip
            cursor={{ fill: "var(--surface-2)" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <TooltipBox title={String(payload[0].payload[xKey])} rows={[{ label, value: format(Number(payload[0].value)) }]} />
              ) : null
            }
          />
          <Bar dataKey={yKey} radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map((row, i) => (
              <Cell key={i} fill={highlightBelow !== undefined && Number(row[yKey]) < highlightBelow ? "var(--series-2)" : "var(--series-1)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
