import axios from "axios";

// Empty NEXT_PUBLIC_API_URL = same-origin (Hugging Face Space). Unset = local Node proxy.
const baseURL = (
  process.env.NEXT_PUBLIC_API_URL !== undefined
    ? process.env.NEXT_PUBLIC_API_URL
    : "http://localhost:5000"
).replace(/\/$/, "");

export const api = axios.create({
  baseURL,
  timeout: 15_000,
  headers: { "Content-Type": "application/json" },
});

export async function fetchRecommendations(userId, n = 10) {
  const { data } = await api.get(
    `/api/recommendations/${encodeURIComponent(userId)}`,
    { params: { n } }
  );
  return data.recommendations ?? [];
}

export async function sendFeedback(userId, videoId, signal) {
  await api.post("/api/feedback", {
    user_id: userId,
    video_id: videoId,
    signal,
  });
}

export async function fetchHealth() {
  const { data } = await api.get("/api/health");
  return data;
}
