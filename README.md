---
title: Clio Recommendations
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Clio

Personalized movie recommendations powered by **ALS collaborative filtering** on [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/). Next.js UI, Flask API, real-time thumbs-up/down feedback.

<p align="center">
  <a href="https://abzyvantae-clio-recommeder.hf.space">Live demo (HF)</a>
  ·
  <a href="https://github.com/abeeraisabeera/Clio-Video-Recommendation-System">GitHub</a>
</p>

<p align="center">
  <img src="./assets/demo.jpg" alt="Clio recommendation UI" width="720" />
</p>

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Next.js 16, React 19, Tailwind 4 |
| API | Flask, `implicit` ALS |
| Data | MovieLens 100K → `clio_interactions.csv` |
| Deploy | [Hugging Face Spaces](https://huggingface.co/spaces/abzyvantae/clio_recommeder) · Vercel |

```
Browser (Next.js)  →  /api/*  →  Flask + ALS  →  MovieLens matrix
```

## Features

- ALS recommendations with title, genre, match score, and thumbnails
- Thumbs up / down reranking without retraining
- Offline metrics: Precision@k, Recall@k, NDCG@k
- Docker Space bundles UI + API on one URL

## Quick start

**Prerequisites:** Python 3.12+, Node 22+, pnpm

```bash
git clone https://github.com/abeeraisabeera/Clio-Video-Recommendation-System.git
cd Clio-Video-Recommendation-System

python prepare_data.py          # once — downloads MovieLens
pip install -r requirements-docker.txt
corepack enable && pnpm install

pnpm dev                        # Flask :5000 + Next.js :3000
```

Copy `.env.example` → `clio-frontend/.env.local` if you need a custom API URL (default `http://localhost:5000`).

```bash
pnpm test                       # pytest + vitest
pnpm evaluate                   # offline evaluation
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Model status |
| GET | `/api/recommendations/:userId?n=10` | Top-N recommendations |
| POST | `/api/feedback` | `{ user_id, video_id, signal: "up"\|"down" }` |
| GET | `/api/movies?limit=100` | Catalogue browse |

```bash
curl "https://abzyvantae-clio-recommeder.hf.space/api/recommendations/user_196?n=5"
```

## Model

| Setting | Value |
|---------|-------|
| Algorithm | ALS (`implicit`) |
| Factors / reg / iterations | 100 / 0.05 / 50 |
| Matrix | 943 users × 1,682 items |
| Inference | `recalculate_user=True` |

Shared training and inference live in `model.py` (used by `app.py`, `evaluate.py`, and tests).

## Deploy

### Hugging Face (API + UI)

This repo is configured for [abzyvantae/clio_recommeder](https://huggingface.co/spaces/abzyvantae/clio_recommeder). Push updates or connect GitHub; the `Dockerfile` serves Flask on port **7860** with a prebuilt static UI.

```bash
pnpm build:hf
Copy-Item -Recurse clio-frontend/out static   # PowerShell
docker build -t clio . && docker run -p 7860:7860 clio
```

### Vercel (frontend only)

1. Import the GitHub repo
2. **Root Directory:** `clio-frontend`
3. **Env:** `NEXT_PUBLIC_API_URL` = `https://abzyvantae-clio-recommeder.hf.space`
4. Redeploy

## Project layout

```
├── app.py              Flask API
├── model.py            ALS training & inference
├── prepare_data.py     MovieLens → CSV
├── evaluate.py         Offline metrics
├── clio-frontend/      Next.js app
├── tests/              pytest
├── Dockerfile          HF Space
└── requirements-docker.txt
```

## License

MIT · Dataset: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (GroupLens).
