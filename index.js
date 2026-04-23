/**
 * Clio Recommendation API — Node.js proxy
 *
 * Routes:
 *   GET  /api/health
 *   GET  /api/recommendations/:userId?n=10
 *   POST /api/feedback
 *   GET  /api/movies?limit=100
 */

const express = require("express");
const axios   = require("axios");
const cors    = require("cors");

const app       = express();
const PORT      = process.env.PORT      || 3001;
const FLASK_BASE = process.env.FLASK_URL || "http://localhost:5000";

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

app.use(cors());
app.use(express.json());

// In-memory rate limiter (swap for express-rate-limit + Redis in production)
const RATE_WINDOW_MS = 60_000;
const RATE_LIMIT     = 60;
const requestCounts  = new Map();

function rateLimiter(req, res, next) {
  const ip  = req.ip || req.connection.remoteAddress;
  const now = Date.now();
  const e   = requestCounts.get(ip) || { count: 0, windowStart: now };
  if (now - e.windowStart > RATE_WINDOW_MS) { e.count = 0; e.windowStart = now; }
  e.count++;
  requestCounts.set(ip, e);
  if (e.count > RATE_LIMIT)
    return res.status(429).json({ error: "rate_limit_exceeded", message: "Too many requests." });
  next();
}

function logger(req, _res, next) {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
}

app.use(logger);
app.use(rateLimiter);

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

const isValidUserId  = (id)  => typeof id === "string" && /^[\w-]{1,64}$/.test(id);
const isValidVideoId = (id)  => typeof id === "string" && id.length > 0 && id.length <= 64;

function parseN(raw) {
  if (raw === undefined) return 10;
  const n = parseInt(raw, 10);
  return Number.isNaN(n) || n < 1 || n > 50 ? null : n;
}

// ---------------------------------------------------------------------------
// Proxy helper
// ---------------------------------------------------------------------------

async function proxyFlask(res, flaskRequest) {
  try {
    const { data, status } = await flaskRequest();
    return res.status(status).json(data);
  } catch (err) {
    if (err.response) {
      const { status, data } = err.response;
      return res.status(status).json({
        error:   data.error   || "upstream_error",
        message: data.message || "The model server returned an error.",
      });
    }
    console.error("Flask unreachable:", err.message);
    return res.status(503).json({
      error:   "service_unavailable",
      message: "The recommendation service is temporarily unavailable.",
    });
  }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/** GET /api/health */
app.get("/api/health", async (_req, res) => {
  await proxyFlask(res, () => axios.get(`${FLASK_BASE}/health`, { timeout: 5000, validateStatus: () => true }));
});

/** GET /api/recommendations/:userId?n=10 */
app.get("/api/recommendations/:userId", async (req, res) => {
  const { userId } = req.params;
  const n = parseN(req.query.n);

  if (!isValidUserId(userId))
    return res.status(400).json({ error: "invalid_user_id", message: "user_id must be 1–64 alphanumeric characters." });
  if (n === null)
    return res.status(400).json({ error: "invalid_n", message: "n must be an integer between 1 and 50." });

  await proxyFlask(res, () =>
    axios.get(`${FLASK_BASE}/recommend`, {
      params: { user_id: userId, n },
      timeout: 10_000,
      validateStatus: () => true,
    })
  );
});

/** POST /api/feedback  { user_id, video_id, signal: "up"|"down" } */
app.post("/api/feedback", async (req, res) => {
  const { user_id, video_id, signal } = req.body || {};

  if (!isValidUserId(user_id))
    return res.status(400).json({ error: "invalid_user_id", message: "user_id is required." });
  if (!isValidVideoId(video_id))
    return res.status(400).json({ error: "invalid_video_id", message: "video_id is required." });
  if (!["up", "down"].includes(signal))
    return res.status(400).json({ error: "invalid_signal", message: "signal must be 'up' or 'down'." });

  await proxyFlask(res, () =>
    axios.post(`${FLASK_BASE}/feedback`, { user_id, video_id, signal }, {
      timeout: 5_000,
      validateStatus: () => true,
    })
  );
});

/** GET /api/movies?limit=100 */
app.get("/api/movies", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit, 10) || 100, 1000);
  await proxyFlask(res, () =>
    axios.get(`${FLASK_BASE}/movies`, { params: { limit }, timeout: 5_000, validateStatus: () => true })
  );
});

// 404 catch-all
app.use((_req, res) => res.status(404).json({ error: "not_found", message: "Endpoint does not exist." }));

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log(`Clio proxy  →  http://localhost:${PORT}`);
  console.log(`Flask target →  ${FLASK_BASE}`);
});
