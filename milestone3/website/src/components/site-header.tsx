"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV, REPO } from "@/lib/site";
import { Logo } from "./logo";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 px-3 pt-3 sm:px-4">
      <div className="glass mx-auto backdrop-blur-xl backdrop-saturate-150 flex h-13 max-w-6xl items-center gap-2 rounded-2xl border border-line px-2.5 shadow-[0_8px_30px_rgba(0,0,0,0.25)] sm:px-4">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="hidden font-semibold tracking-tight min-[420px]:inline">Lens 73</span>
          <span className="hidden text-xs text-muted md:inline">Group 1 · IE 7615</span>
        </Link>
        <nav className="ml-auto flex items-center text-[13px] sm:gap-0.5 sm:text-sm">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                className={`rounded-lg px-2 py-1.5 transition-colors sm:px-2.5 ${active ? "bg-surface-3/70 text-ink" : "text-ink-2 hover:text-ink"}`}
              >
                {n.label}
              </Link>
            );
          })}
          <a href={REPO} target="_blank" rel="noreferrer" className="hidden rounded-lg px-2.5 py-1.5 text-ink-2 hover:text-ink sm:inline">
            GitHub
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
