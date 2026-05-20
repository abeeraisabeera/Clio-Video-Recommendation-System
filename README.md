---
title: Clio Recommendations
emoji: 🎬
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Clio — Video Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-13%2B-black)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A full-stack recommendation system using **Alternating Least Squares (ALS)** collaborative filtering on the MovieLens 100K dataset. Built to demonstrate end-to-end ML system design: data pipeline → model training → REST API → interactive frontend with real-time feedback.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser                            │
│              Next.js  (localhost:3000)                  │
│         Search · Cards · Thumbs up/down UI              │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│         Flask API  (localhost:5000 / HF :7860)            │
│  ALS · /api/recommendations · feedback · health          │
└──────────────────────┬──────────────────────────────────┘
                       │
              clio_interactions.csv
              (MovieLens 100K, 100,000 ratings)
```

---

## Features

- **Collaborative filtering** via ALS (`implicit` library) on 100K real user–item interactions
- **Rich recommendations** — each result includes title, category, match score, thumbnail, duration
- **Feedback loop** — thumbs up/down adjusts scores in real time without retraining
- **Offline evaluation** — Precision@k, Recall@k, NDCG@k with train/test split
- **Production patterns** — health endpoint, rate limiting, structured errors, request logging
- **Skeleton loaders**, staggered animations, responsive card grid

---

## Quickstart

### 1. Prepare data

```bash
python prepare_data.py
```

Downloads MovieLens 100K (~5 MB) and outputs:
- `clio_interactions.csv` — 100,000 user–movie interactions with weights
- `movies.csv` — metadata for 1,682 movies

### 2. Python backend

```bash
python -m venv venv && source venv/bin/activate
pip install flask flask-cors implicit scikit-learn scipy pandas numpy
python app.py
# → http://localhost:5000
```

### 3. Install JS dependencies (pnpm)

```bash
corepack enable
pnpm install
```

### 4. API + frontend

```bash
# Flask (port 5000) + Next.js (port 3000)
pnpm dev
```

Copy `.env.example` to `clio-frontend/.env.local`. The UI calls `NEXT_PUBLIC_API_URL` (default `http://localhost:5000`).

### 5. Tests & evaluation

```bash
pnpm test              # pytest + vitest
pnpm evaluate          # Precision@k, Recall@k, NDCG@k
python evaluate.py --users 500 --k 10
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Liveness probe — model stats |
| `GET`  | `/api/recommendations/:userId?n=10` | Ranked recommendations with metadata |
| `POST` | `/api/feedback` | Record thumbs up / down |
| `GET`  | `/api/movies?limit=100` | Browse full catalogue |

### Example — get recommendations

```bash
curl "http://localhost:5000/api/recommendations/user_196?n=5"
```

```json
{
  "user_id": "user_196",
  "recommendations": [
    {
      "video_id": "movie_50",
      "score": 0.9821,
      "title": "Star Wars (1977)",
      "category": "Action",
      "thumbnail_url": "https://picsum.photos/seed/movie_50/320/180",
      "duration_seconds": 7680
    }
  ]
}
```

### Example — send feedback

```bash
curl -X POST http://localhost:5000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_196", "video_id": "movie_50", "signal": "up"}'
```

---
## Evaluation

Run offline metrics against a held-out test split:

```bash
python evaluate.py
```

```
====================================================
  Clio ALS Recommender — Evaluation Report
  Dataset : clio_interactions.csv
  Split   : 80/20 train/test
  Users   : 300 sampled
====================================================
     k   Precision@k    Recall@k    NDCG@k
  ----------------------------------------
     5        0.2913      0.1097    0.2833
    10        0.2483      0.1807    0.2771
    20        0.1985      0.2730    0.2930
====================================================
```

Options:

```bash
python evaluate.py --k 20 --users 500 --split 0.2
```

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | ALS (Alternating Least Squares) |
| Library | `implicit` |
| Latent factors | 100 |
| Regularization | 0.05 |
| Iterations | 50 |
| Inference | `recalculate_user=True` |
| Signal | 1 + 2.5 × max(rating − 2.5, 0) |

The interaction matrix is sparse (`scipy.csr_matrix`, 943 users × 1,682 items). ALS decomposes it into user and item factor matrices, enabling O(1) inference per recommendation request.

---

## Demo ![Demo](./assets/demo.jpg)

---
## Project Structure

```
clio Recommender/
├── prepare_data.py      # Downloads MovieLens, generates CSVs
├── app.py               # Flask ML server
├── evaluate.py          # Offline metrics (Precision, Recall, NDCG)
├── model.py             # Shared ALS training & inference
├── tests/               # pytest (API + model)
├── clio-frontend/
│   ├── app/             # Next.js App Router
│   ├── components/      # UI components
│   └── lib/             # API client & helpers
├── Dockerfile           # Hugging Face Spaces (Docker SDK)
├── pnpm-workspace.yaml
├── pnpm-lock.yaml
├── clio_interactions.csv  # Generated by prepare_data.py
└── movies.csv             # Generated by prepare_data.py
```

---

## Deploy (live) — Hugging Face Spaces

Everything runs in **one Docker Space**: static Next.js UI + Flask ALS API on port `7860`. No Render or separate frontend host required.

### 1. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK.
3. Connect this GitHub repo (or push a copy to a HF git remote).

The `README.md` frontmatter and root `Dockerfile` are already configured for HF.

### 2. Hardware

- Start with **CPU basic** (free). First boot trains ALS (~30–90s) then serves traffic.
- Upgrade to **CPU upgrade** if cold starts feel slow.

### 3. Environment variables (optional)

| Variable | Default | Purpose |
|----------|---------|---------|
| `STATIC_ROOT` | `/app/static` | Built Next.js export inside the container |
| `PORT` | `7860` | HF-required listen port |

No `NEXT_PUBLIC_API_URL` needed in production — the UI calls `/api/*` on the same origin.

### 4. Local Docker smoke test

```bash
docker build -t clio .
docker run -p 7860:7860 clio
# → http://localhost:7860
```

### Local dev

```bash
pnpm dev
# or Flask-only with built static UI:
pnpm build:hf
set STATIC_ROOT=clio-frontend\out
python app.py
```

---

## Production Roadmap

- [ ] Persist feedback to PostgreSQL and schedule nightly retrains
- [ ] Add rate limiting on Flask for public deployments
- [ ] Add item-based fallback for cold-start users
- [x] Docker image for Hugging Face Spaces (UI + API)

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TIIS 5(4).
