# Hugging Face Spaces Docker image — UI + ALS API on port 7860
# https://huggingface.co/docs/hub/spaces-sdks-docker

# --- Build Next.js static export ---
FROM node:22-bookworm-slim AS frontend
WORKDIR /app

RUN corepack enable

COPY package.json pnpm-workspace.yaml pnpm-lock.yaml .npmrc ./
COPY clio-frontend/package.json ./clio-frontend/

RUN pnpm install --frozen-lockfile

COPY clio-frontend ./clio-frontend

ENV STATIC_EXPORT=1
ENV NEXT_PUBLIC_API_URL=

RUN pnpm --filter clio-frontend build

# --- Python runtime ---
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py prepare_data.py evaluate.py ./
COPY clio_interactions.csv movies.csv ./
COPY --from=frontend /app/clio-frontend/out ./static

ENV STATIC_ROOT=/app/static
ENV PORT=7860

EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 2 --timeout 180 app:app
