import { CountUp, Reveal } from "./motion";

export function Section({ title, kicker, id, children }: { title: string; kicker?: string; id?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mt-20 scroll-mt-24">
      <Reveal>
        {kicker && <p className="text-sm font-medium text-brand">{kicker}</p>}
        <h2 className="mt-1 text-2xl font-semibold tracking-tight sm:text-4xl">{title}</h2>
      </Reveal>
      <div className="mt-6">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <Reveal y={20} className="rounded-2xl border border-line bg-surface p-5 transition-colors hover:border-line-strong">
      <p className="text-xs text-ink-2">{label}</p>
      <p className="tabular mt-1.5 text-3xl font-semibold tracking-tight">{value.includes("/") ? value : <CountUp value={value} />}</p>
      {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
    </Reveal>
  );
}

export function Figure({ title, caption, className = "", children }: { title: string; caption?: string; className?: string; children: React.ReactNode }) {
  return (
    <Reveal as="figure" y={32} className={`rounded-2xl border border-line bg-surface p-5 ${className}`}>
      <figcaption className="mb-4">
        <p className="text-sm font-semibold">{title}</p>
        {caption && <p className="mt-1 text-xs leading-relaxed text-ink-2">{caption}</p>}
      </figcaption>
      {children}
    </Reveal>
  );
}

export function PageHeader({ kicker, title, children }: { kicker: string; title: React.ReactNode; children?: React.ReactNode }) {
  return (
    <header className="fade-up relative pt-14">
      <div className="aurora pointer-events-none absolute left-1/2 -top-24 h-80 w-screen -translate-x-1/2 opacity-80" />
      <p className="relative text-sm font-medium text-brand">{kicker}</p>
      <h1 className="relative mt-2 text-4xl font-semibold tracking-[-0.035em] sm:text-6xl">{title}</h1>
      {children && <div className="relative mt-4 max-w-2xl text-lg text-ink-2">{children}</div>}
    </header>
  );
}
