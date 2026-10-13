import { REPO, REPORTS, TEAM } from "@/lib/site";
import { Logo } from "./logo";

export function SiteFooter() {
  return (
    <footer className="mt-28 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-12 text-sm sm:grid-cols-[1.3fr_1fr_1fr] sm:px-6">
        <div>
          <div className="flex items-center gap-2.5">
            <Logo />
            <span className="font-semibold tracking-tight">Lens 73</span>
          </div>
          <p className="mt-3 max-w-xs text-ink-2">
            IE 7615 Discriminative Deep Learning · Northeastern University · Fall 2026. Both models run in your browser; photos are not uploaded.
          </p>
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted">Group 1</p>
          <ul className="mt-3 space-y-1.5 text-ink-2">
            {TEAM.map((m) => (
              <li key={m.nuid}>
                {m.name} <span className="tabular text-muted">{m.nuid}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted">Resources</p>
          <ul className="mt-3 space-y-1.5">
            <li>
              <a className="text-ink-2 hover:text-ink" href={REPO} target="_blank" rel="noreferrer">GitHub repository</a>
            </li>
            {REPORTS.map((r) => (
              <li key={r.href}>
                <a className="text-ink-2 hover:text-ink" href={r.href} target="_blank" rel="noreferrer">{r.label}</a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </footer>
  );
}
