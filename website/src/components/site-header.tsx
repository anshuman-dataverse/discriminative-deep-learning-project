"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV, REPO } from "@/lib/site";
import { ThemeToggle } from "./theme-toggle";

export function SiteHeader() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-page/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-brand text-xs font-bold text-white">G1</span>
          <span className="hidden sm:inline">Object ID &amp; Detection</span>
        </Link>
        <nav className="ml-auto flex items-center gap-1 text-sm">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                className={`rounded-md px-2.5 py-1.5 transition-colors ${
                  active ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"
                }`}
              >
                {n.label}
              </Link>
            );
          })}
          <a href={REPO} target="_blank" rel="noreferrer" className="hidden rounded-md px-2.5 py-1.5 text-ink-2 hover:text-ink sm:inline">
            GitHub
          </a>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
