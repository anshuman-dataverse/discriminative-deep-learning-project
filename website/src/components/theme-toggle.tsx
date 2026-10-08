"use client";

import { Moon, Sun } from "lucide-react";

export function ThemeToggle() {
  const toggle = () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {}
  };
  return (
    <button onClick={toggle} aria-label="Switch light or dark theme" className="grid h-8 w-8 shrink-0 sm:ml-1 place-items-center rounded-lg text-ink-2 hover:bg-surface-3/70 hover:text-ink">
      <Sun className="h-4 w-4 dark:block hidden" />
      <Moon className="h-4 w-4 dark:hidden" />
    </button>
  );
}
