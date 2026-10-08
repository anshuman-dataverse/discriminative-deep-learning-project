export function Section({ title, kicker, id, children }: { title: string; kicker?: string; id?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="mt-14 scroll-mt-20">
      {kicker && <p className="text-xs font-semibold uppercase tracking-wider text-muted">{kicker}</p>}
      <h2 className="mt-1 text-2xl font-semibold tracking-tight">{title}</h2>
      <div className="mt-5">{children}</div>
    </section>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <p className="text-xs text-ink-2">{label}</p>
      <p className="mt-1 text-2xl font-semibold tracking-tight">{value}</p>
      {sub && <p className="mt-0.5 text-xs text-muted">{sub}</p>}
    </div>
  );
}

export function Figure({ title, caption, className = "", children }: { title: string; caption?: string; className?: string; children: React.ReactNode }) {
  return (
    <figure className={`rounded-xl border border-line bg-surface p-4 ${className}`}>
      <figcaption className="mb-3">
        <p className="text-sm font-semibold">{title}</p>
        {caption && <p className="mt-0.5 text-xs text-ink-2">{caption}</p>}
      </figcaption>
      {children}
    </figure>
  );
}
