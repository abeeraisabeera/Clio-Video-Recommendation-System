"use client";

import { RefreshCw, Users } from "lucide-react";

const SAMPLE_USERS = ["user_1", "user_2", "user_3", "user_196", "user_405"];

export function Controls({
  userId,
  count,
  loading,
  onUserSelect,
  onCountChange,
  onRefresh,
}) {
  return (
    <section className="mx-auto max-w-6xl px-4 pt-6 sm:px-6">
      <div className="glass flex flex-col gap-6 rounded-xl p-4 sm:flex-row sm:items-end sm:justify-between sm:p-5">
        <div className="space-y-3">
          <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-[var(--color-muted)]">
            <Users className="size-3.5" />
            Quick switch
          </p>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_USERS.map((uid) => (
              <button
                key={uid}
                type="button"
                onClick={() => onUserSelect(uid)}
                className={`rounded-full border px-3 py-1.5 text-xs font-medium transition ${
                  userId === uid
                    ? "border-[var(--color-accent-glow)] bg-[var(--color-accent-dim)] text-cyan-200"
                    : "border-[var(--color-border)] bg-transparent text-[var(--color-muted)] hover:border-[var(--color-border-strong)] hover:text-[var(--color-fg)]"
                }`}
              >
                {uid}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-[140px]">
            <p className="mb-2 text-xs text-[var(--color-muted)]">
              Results: <span className="font-semibold text-[var(--color-fg)]">{count}</span>
            </p>
            <input
              type="range"
              min={5}
              max={50}
              step={5}
              value={count}
              onChange={(e) => onCountChange(Number(e.target.value))}
              className="h-1.5 w-full cursor-pointer accent-[var(--color-accent)]"
              aria-label="Number of results"
            />
          </div>
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg border border-[var(--color-border)] px-3 py-2 text-sm text-[var(--color-muted)] transition hover:border-[var(--color-border-strong)] hover:text-[var(--color-fg)] disabled:opacity-40"
          >
            <RefreshCw className={`size-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </div>
    </section>
  );
}
