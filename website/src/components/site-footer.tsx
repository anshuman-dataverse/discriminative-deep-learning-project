import { REPO, REPORTS, TEAM } from "@/lib/site";

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-line">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 text-sm sm:grid-cols-3 sm:px-6">
        <div>
          <p className="font-semibold">Group 1</p>
          <ul className="mt-2 space-y-1 text-ink-2">
            {TEAM.map((m) => (
              <li key={m.nuid}>
                {m.name} <span className="tabular text-muted">({m.nuid})</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="font-semibold">IE 7615 · Discriminative Deep Learning</p>
          <p className="mt-2 text-ink-2">Northeastern University · Fall 2026</p>
          <p className="mt-1 text-ink-2">Models run entirely in your browser (ONNX Runtime Web). No image leaves your device.</p>
        </div>
        <div>
          <p className="font-semibold">Resources</p>
          <ul className="mt-2 space-y-1">
            <li>
              <a className="text-ink-2 underline-offset-4 hover:text-ink hover:underline" href={REPO} target="_blank" rel="noreferrer">
                GitHub repository
              </a>
            </li>
            {REPORTS.map((r) => (
              <li key={r.href}>
                <a className="text-ink-2 underline-offset-4 hover:text-ink hover:underline" href={r.href} target="_blank" rel="noreferrer">
                  {r.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </footer>
  );
}
