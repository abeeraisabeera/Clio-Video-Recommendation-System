# Hugging Face Spaces — Clio ALS API (+ optional prebuilt static UI in ./static)
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py prepare_data.py ./
COPY clio_interactions.csv movies.csv ./

# Optional: add prebuilt Next export to ./static before deploy (pnpm build:hf)
COPY static ./static

ENV STATIC_ROOT=/app/static
ENV PORT=7860

EXPOSE 7860

CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 2 --timeout 180 app:app
