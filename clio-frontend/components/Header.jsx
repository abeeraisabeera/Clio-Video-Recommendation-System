"use client";

import { Search, Sparkles } from "lucide-react";

export function Header({ inputValue, onInputChange, onSearch }) {
  return (
    <header className="glass sticky top-0 z-50 border-b border-[var(--color-border)]">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 items-center justify-center rounded-lg bg-[var(--color-accent-dim)] ring-1 ring-[var(--color-accent-glow)]">
            <Sparkles className="size-4 text-[var(--color-accent)]" strokeWidth={2} />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">
              <span className="text-gradient">Clio</span>
            </h1>
            <p className="hidden text-xs text-[var(--color-muted)] sm:block">
              ALS collaborative filtering
            </p>
          </div>
        </div>

        <form onSubmit={onSearch} className="flex flex-1 max-w-xs sm:max-w-sm gap-2">
          <label className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--color-muted)]" />
            <input
              value={inputValue}
              onChange={(e) => onInputChange(e.target.value)}
              placeholder="User ID…"
              aria-label="User ID"
              className="w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-overlay)] py-2 pl-9 pr-3 text-sm text-[var(--color-fg)] outline-none transition placeholder:text-zinc-500 focus:border-[var(--color-accent-glow)] focus:ring-2 focus:ring-[var(--color-accent-dim)]"
            />
          </label>
          <button
            type="submit"
            className="shrink-0 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-zinc-950 transition hover:brightness-110 active:scale-[0.98]"
          >
            Go
          </button>
        </form>
      </div>
    </header>
  );
}
