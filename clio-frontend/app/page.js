"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertCircle, Film } from "lucide-react";
import { Header } from "@/components/Header";
import { Controls } from "@/components/Controls";
import { VideoCard } from "@/components/VideoCard";
import { SkeletonCard } from "@/components/SkeletonCard";
import { fetchRecommendations } from "@/lib/api";
import { normalizeScores } from "@/lib/utils";

export default function HomePage() {
  const [userId, setUserId] = useState("user_1");
  const [inputValue, setInputValue] = useState("user_1");
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [count, setCount] = useState(10);

  const load = useCallback(async (uid, n) => {
    setLoading(true);
    setError(null);
    try {
      const recs = await fetchRecommendations(uid, n);
      setVideos(normalizeScores(recs));
    } catch (err) {
      setError(
        err.response?.data?.message ||
          err.response?.data?.error ||
          "Could not reach the recommendation API. Start the proxy and Flask server, or check NEXT_PUBLIC_API_URL."
      );
      setVideos([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(userId, count);
  }, [userId, count, load]);

  const handleDismiss = useCallback((videoId) => {
    setTimeout(
      () => setVideos((prev) => prev.filter((v) => v.video_id !== videoId)),
      400
    );
  }, []);

  function handleSearch(e) {
    e.preventDefault();
    const t = inputValue.trim();
    if (t) setUserId(t);
  }

  return (
    <div className="pb-20">
      <Header
        inputValue={inputValue}
        onInputChange={setInputValue}
        onSearch={handleSearch}
      />

      <Controls
        userId={userId}
        count={count}
        loading={loading}
        onUserSelect={(uid) => {
          setUserId(uid);
          setInputValue(uid);
        }}
        onCountChange={setCount}
        onRefresh={() => load(userId, count)}
      />

      <section className="mx-auto max-w-6xl px-4 pt-8 sm:px-6">
        <div className="mb-6 flex items-baseline justify-between gap-4">
          <h2 className="text-xl font-semibold tracking-tight sm:text-2xl">
            For{" "}
            <span className="font-mono text-[var(--color-accent)]">{userId}</span>
          </h2>
          {!loading && videos.length > 0 && (
            <span className="text-sm text-[var(--color-muted)]">
              {videos.length} titles
            </span>
          )}
        </div>

        {error && !loading && (
          <div
            role="alert"
            className="mb-6 flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200"
          >
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4 xs:grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:gap-5">
          {loading
            ? Array.from({ length: Math.min(count, 12) }).map((_, i) => (
                <SkeletonCard key={i} />
              ))
            : videos.map((v, i) => (
                <div
                  key={v.video_id}
                  className="animate-rise"
                  style={{ animationDelay: `${i * 40}ms` }}
                >
                  <VideoCard
                    video={v}
                    rank={i + 1}
                    userId={userId}
                    onDismiss={handleDismiss}
                  />
                </div>
              ))}
        </div>

        {!loading && !error && videos.length === 0 && (
          <div className="flex flex-col items-center gap-3 py-24 text-center">
            <Film className="size-10 text-zinc-600" strokeWidth={1.25} />
            <p className="text-sm text-[var(--color-muted)]">
              No recommendations for this user.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
