"""
Shared ALS model training and inference for Clio.
Used by app.py, evaluate.py, and tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

threadpool_limits(1, "blas")

# Tuned on MovieLens 100K hold-out (evaluate.py)
ALS_FACTORS = 100
ALS_REGULARIZATION = 0.05
ALS_ITERATIONS = 50
BM25_K1 = 100.0
BM25_B = 0.8
USE_BM25 = False

USER_ID_RE = r"^[\w-]{1,64}$"


@dataclass
class ClioModel:
    model: AlternatingLeastSquares
    matrix: sparse.csr_matrix
    user_encoder: LabelEncoder
    video_encoder: LabelEncoder
    video_metadata: dict[str, dict]
    feedback: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def num_users(self) -> int:
        return self.matrix.shape[0]

    @property
    def num_videos(self) -> int:
        return self.matrix.shape[1]


def load_interactions(csv_path: str = "clio_interactions.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"user_id", "video_id", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    meta_cols = [c for c in ("title", "category", "thumbnail_url", "duration_seconds") if c in df.columns]
    weights = df.groupby(["user_id", "video_id"], as_index=False)["weight"].sum()

    if not meta_cols:
        return weights

    meta = df[["video_id", *meta_cols]].drop_duplicates("video_id")
    return weights.merge(meta, on="video_id", how="left")


def _build_video_metadata(df: pd.DataFrame, video_encoder: LabelEncoder) -> dict[str, dict]:
    meta_cols = {"title", "category", "thumbnail_url", "duration_seconds"}
    available = meta_cols & set(df.columns)
    video_meta: dict[str, dict] = {}

    if available:
        vid_meta = df.drop_duplicates("video_id").set_index("video_id")
        for vid in video_encoder.classes_:
            if vid in vid_meta.index:
                row = vid_meta.loc[vid]
                video_meta[str(vid)] = {
                    "title": str(row["title"]) if "title" in available else f"Video {vid}",
                    "category": str(row["category"])
                    if "category" in available
                    else "General",
                    "thumbnail_url": str(row["thumbnail_url"])
                    if "thumbnail_url" in available
                    else f"https://picsum.photos/seed/{vid}/320/180",
                    "duration_seconds": int(row["duration_seconds"])
                    if "duration_seconds" in available
                    and pd.notna(row["duration_seconds"])
                    else None,
                }
            else:
                video_meta[str(vid)] = _default_meta(vid)
    else:
        for vid in video_encoder.classes_:
            video_meta[str(vid)] = _default_meta(vid)

    return video_meta


def _default_meta(vid: str) -> dict:
    return {
        "title": f"Video {vid}",
        "category": "General",
        "thumbnail_url": f"https://picsum.photos/seed/{vid}/320/180",
        "duration_seconds": None,
    }


def build_model(
    csv_path: str = "clio_interactions.csv",
    *,
    use_bm25: bool = USE_BM25,
    logger=None,
) -> ClioModel:
    import logging

    log = logger or logging.getLogger(__name__)

    log.info("Loading interactions from %s", csv_path)
    df = load_interactions(csv_path)

    user_enc = LabelEncoder()
    video_enc = LabelEncoder()
    df = df.copy()
    df["user_index"] = user_enc.fit_transform(df["user_id"])
    df["video_index"] = video_enc.fit_transform(df["video_id"])

    matrix = sparse.csr_matrix(
        (df["weight"].astype(float), (df["user_index"], df["video_index"])),
        shape=(len(user_enc.classes_), len(video_enc.classes_)),
    )

    if use_bm25:
        log.info("Applying BM25 weighting")
        matrix = bm25_weight(matrix, K1=BM25_K1, B=BM25_B).tocsr()

    log.info("Training ALS — %d users × %d items", *matrix.shape)
    t0 = time.time()
    als = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=42,
    )
    als.fit(matrix)
    log.info("Trained in %.2fs", time.time() - t0)

    video_meta = _build_video_metadata(df, video_enc)
    return ClioModel(
        model=als,
        matrix=matrix,
        user_encoder=user_enc,
        video_encoder=video_enc,
        video_metadata=video_meta,
    )


def build_model_from_frame(
    train: pd.DataFrame,
    *,
    use_bm25: bool = USE_BM25,
) -> tuple[AlternatingLeastSquares, sparse.csr_matrix, LabelEncoder, LabelEncoder]:
    """Train on a DataFrame slice (for offline evaluation)."""
    train = train.copy()
    user_enc = LabelEncoder()
    video_enc = LabelEncoder()
    train["user_index"] = user_enc.fit_transform(train["user_id"])
    train["video_index"] = video_enc.fit_transform(train["video_id"])

    matrix = sparse.csr_matrix(
        (train["weight"].astype(float), (train["user_index"], train["video_index"])),
        shape=(len(user_enc.classes_), len(video_enc.classes_)),
    )

    if use_bm25:
        matrix = bm25_weight(matrix, K1=BM25_K1, B=BM25_B).tocsr()

    als = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        random_state=42,
    )
    als.fit(matrix)
    return als, matrix, user_enc, video_enc


def recommend_for_user(clio: ClioModel, user_id: str, n: int) -> dict:
    try:
        user_idx = clio.user_encoder.transform([user_id])[0]
    except ValueError as err:
        raise LookupError(f"Unknown user_id '{user_id}'") from err

    fetch_n = min(n + 30, clio.num_videos)
    ids, scores = clio.model.recommend(
        user_idx,
        clio.matrix[user_idx],
        N=fetch_n,
        filter_already_liked_items=True,
        recalculate_user=True,
    )
    video_ids = clio.video_encoder.inverse_transform(ids).tolist()

    fb = clio.feedback.get(user_id, {})
    adjusted = [
        (vid, float(scores[i]) + fb.get(vid, 0.0)) for i, vid in enumerate(video_ids)
    ]
    adjusted = [(v, s) for v, s in adjusted if fb.get(v, 0.0) >= 0]
    adjusted.sort(key=lambda x: x[1], reverse=True)
    adjusted = adjusted[:n]

    recommendations = [
        {
            "video_id": vid,
            "score": round(score, 4),
            **clio.video_metadata.get(vid, _default_meta(vid)),
        }
        for vid, score in adjusted
    ]
    return {"user_id": user_id, "recommendations": recommendations}


def record_feedback(clio: ClioModel, user_id: str, video_id: str, signal: str) -> dict:
    delta = 0.3 if signal == "up" else -1.0
    clio.feedback.setdefault(user_id, {})[video_id] = delta
    return {
        "status": "recorded",
        "user_id": user_id,
        "video_id": video_id,
        "signal": signal,
    }
