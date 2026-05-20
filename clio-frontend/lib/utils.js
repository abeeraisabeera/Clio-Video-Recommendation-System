export function formatDuration(seconds) {
  if (!seconds) return null;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function normalizeScores(videos) {
  if (!videos.length) return [];
  const scores = videos.map((v) => v.score);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  return videos.map((v) => ({
    ...v,
    norm: max > min ? (v.score - min) / (max - min) : 0,
  }));
}
