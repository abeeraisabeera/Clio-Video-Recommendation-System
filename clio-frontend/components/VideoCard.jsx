"use client";

import { useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { sendFeedback } from "@/lib/api";
import { formatDuration } from "@/lib/utils";

export function VideoCard({ video, rank, userId, onDismiss }) {
  const [feedback, setFeedback] = useState(null);
  const [sending, setSending] = useState(false);

  async function handleFeedback(signal) {
    if (sending || feedback) return;
    setSending(true);
    try {
      await sendFeedback(userId, video.video_id, signal);
      setFeedback(signal);
      if (signal === "down") onDismiss(video.video_id);
    } catch {
      /* ignore */
    } finally {
      setSending(false);
    }
  }

  const relevance = Math.round((video.norm ?? 0) * 100);

  return (
    <article
      className={`group glass overflow-hidden rounded-xl transition duration-300 hover:-translate-y-1 hover:shadow-[0_20px_50px_-12px_rgba(0,0,0,0.5)] hover:ring-1 hover:ring-[var(--color-accent-glow)] ${
        feedback === "down" ? "pointer-events-none opacity-0 scale-95" : ""
      }`}
    >
      <div className="relative aspect-video overflow-hidden bg-zinc-900">
        <img
          src={
            video.thumbnail_url ||
            `https://picsum.photos/seed/${video.video_id}/640/360`
          }
          alt={video.title}
          loading="lazy"
          className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.03]"
        />
        <span className="absolute left-2 top-2 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] font-bold tabular-nums text-zinc-300 backdrop-blur-sm">
          #{rank}
        </span>
        {video.duration_seconds && (
          <span className="absolute bottom-2 right-2 rounded-md bg-black/75 px-1.5 py-0.5 text-[10px] font-semibold text-white backdrop-blur-sm">
            {formatDuration(video.duration_seconds)}
          </span>
        )}
        <div
          className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/80 to-transparent opacity-0 transition group-hover:opacity-100"
          aria-hidden
        />
      </div>

      <div className="space-y-3 p-3.5">
        <h3 className="line-clamp-2 text-sm font-medium leading-snug text-[var(--color-fg)]">
          {video.title}
        </h3>

        <div className="flex flex-wrap items-center gap-2">
          {video.category && (
            <span className="rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-medium capitalize text-zinc-400">
              {video.category}
            </span>
          )}
          {video.score !== undefined && (
            <span className="rounded-md bg-[var(--color-accent-dim)] px-2 py-0.5 text-[10px] font-semibold text-cyan-300/90">
              {relevance}% match
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 border-t border-[var(--color-border)] pt-3">
          <button
            type="button"
            onClick={() => handleFeedback("up")}
            disabled={!!feedback || sending}
            aria-label="More like this"
            className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs transition disabled:opacity-40 ${
              feedback === "up"
                ? "border-emerald-500/40 bg-emerald-500/15 text-[var(--color-success)]"
                : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
            }`}
          >
            <ThumbsUp className="size-3.5" />
          </button>
          <button
            type="button"
            onClick={() => handleFeedback("down")}
            disabled={!!feedback || sending}
            aria-label="Not interested"
            className={`inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs transition disabled:opacity-40 ${
              feedback === "down"
                ? "border-red-500/40 bg-red-500/15 text-[var(--color-danger)]"
                : "border-[var(--color-border)] hover:border-[var(--color-border-strong)]"
            }`}
          >
            <ThumbsDown className="size-3.5" />
          </button>
          {feedback === "up" && (
            <span className="text-[10px] text-[var(--color-success)]">Noted</span>
          )}
        </div>
      </div>
    </article>
  );
}
