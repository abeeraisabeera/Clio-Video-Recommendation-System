export function SkeletonCard() {
  return (
    <div className="glass overflow-hidden rounded-xl">
      <div className="aspect-video animate-shimmer bg-zinc-800/60" />
      <div className="space-y-2 p-3.5">
        <div className="h-3.5 w-4/5 animate-shimmer rounded bg-zinc-800/60" />
        <div className="h-3 w-1/3 animate-shimmer rounded bg-zinc-800/40" />
      </div>
    </div>
  );
}
