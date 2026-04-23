
"""
app.py — Clio Recommendation API (Flask)

Endpoints:
    GET  /health                          liveness probe
    GET  /recommend?user_id=&n=           personalised recommendations
    POST /feedback                        thumbs-up / thumbs-down
    GET  /movies                          full catalogue (for browsing)
"""

import logging
import time
import pandas as pd
import scipy.sparse as sparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

threadpool_limits(1, "blas")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(csv_path: str = "clio_interactions.csv"):
    logger.info("Loading interactions from %s", csv_path)
    df = pd.read_csv(csv_path)

    required = {"user_id", "video_id", "weight"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required}")

    user_enc  = LabelEncoder()
    video_enc = LabelEncoder()
    df["user_index"]  = user_enc.fit_transform(df["user_id"])
    df["video_index"] = video_enc.fit_transform(df["video_id"])

    matrix = sparse.csr_matrix(
        (df["weight"].astype(float),
         (df["user_index"], df["video_index"])),
        shape=(len(user_enc.classes_), len(video_enc.classes_)),
    )

    logger.info("Training ALS — %d users × %d items", *matrix.shape)
    t0 = time.time()
    model = AlternatingLeastSquares(factors=100, regularization=0.05, iterations=50)
    model.fit(matrix)
    logger.info("Trained in %.2fs", time.time() - t0)

    # Build video metadata lookup
    meta_cols = {"title", "category", "thumbnail_url", "duration_seconds"}
    available = meta_cols & set(df.columns)
    video_meta: dict[str, dict] = {}
    for vid in video_enc.classes_:
        rows = df[df["video_id"] == vid]
        row  = rows.iloc[0] if len(rows) else None
        video_meta[vid] = {
            "title":            str(row["title"])          if row is not None and "title"            in available else f"Video {vid}",
            "category":         str(row["category"])       if row is not None and "category"         in available else "General",
            "thumbnail_url":    str(row["thumbnail_url"])  if row is not None and "thumbnail_url"    in available else f"https://picsum.photos/seed/{vid}/320/180",
            "duration_seconds": int(row["duration_seconds"]) if row is not None and "duration_seconds" in available else None,
        }

    return model, matrix, user_enc, video_enc, video_meta, df


try:
    model, interaction_matrix, user_encoder, video_encoder, video_metadata, base_df = build_model()
except Exception as exc:
    logger.exception("Model init failed: %s", exc)
    raise SystemExit(1) from exc

# In-memory feedback store  {user_id: {video_id: delta_weight}}
# In production: persist to a DB and periodically retrain.
feedback_store: dict[str, dict[str, float]] = {}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status":      "ok",
        "model":       "ALS (implicit)",
        "num_users":   interaction_matrix.shape[0],
        "num_videos":  interaction_matrix.shape[1],
    })


@app.route("/recommend")
def recommend():
    """
    GET /recommend?user_id=user_196&n=10

    Returns ranked video recommendations with metadata and match scores.
    Feedback adjustments are applied on top of the base model scores.
    """
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    try:
        n = max(1, min(int(request.args.get("n", 10)), 50))
    except ValueError:
        return jsonify({"error": "n must be an integer"}), 400

    try:
        user_idx = user_encoder.transform([user_id])[0]
    except ValueError:
        return jsonify({"error": f"Unknown user_id '{user_id}'"}), 404

    try:
        ids, scores = model.recommend(
            user_idx,
            interaction_matrix[user_idx],
            N=n + 20,                        # over-fetch so feedback can re-rank
            filter_already_liked_items=True,
        )
        video_ids = video_encoder.inverse_transform(ids).tolist()

        # Apply feedback adjustments
        fb = feedback_store.get(user_id, {})
        adjusted = [
            (vid, float(scores[i]) + fb.get(vid, 0.0))
            for i, vid in enumerate(video_ids)
        ]
        # Filter thumbs-down (negative feedback), sort, take N
        adjusted = [(v, s) for v, s in adjusted if fb.get(v, 0.0) >= 0]
        adjusted.sort(key=lambda x: x[1], reverse=True)
        adjusted = adjusted[:n]

        recommendations = [
            {
                "video_id": vid,
                "score":    round(score, 4),
                **video_metadata.get(vid, {}),
            }
            for vid, score in adjusted
        ]

        logger.info("Served %d recs for %s", len(recommendations), user_id)
        return jsonify({"user_id": user_id, "recommendations": recommendations})

    except Exception as exc:
        logger.exception("Recommend error for %s: %s", user_id, exc)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    POST /feedback
    Body: { "user_id": "user_196", "video_id": "movie_242", "signal": "up" | "down" }

    Stores lightweight feedback that adjusts future recommendation scores
    without requiring a full model retrain.
    """
    body = request.get_json(silent=True) or {}
    user_id  = str(body.get("user_id",  "")).strip()
    video_id = str(body.get("video_id", "")).strip()
    signal   = str(body.get("signal",   "")).strip().lower()

    if not user_id or not video_id:
        return jsonify({"error": "user_id and video_id are required"}), 400
    if signal not in ("up", "down"):
        return jsonify({"error": "signal must be 'up' or 'down'"}), 400

    delta = +0.3 if signal == "up" else -1.0     # down buries the item
    feedback_store.setdefault(user_id, {})[video_id] = delta

    logger.info("Feedback: %s → %s (%s)", user_id, video_id, signal)
    return jsonify({"status": "recorded", "user_id": user_id, "video_id": video_id, "signal": signal})


@app.route("/movies")
def movies():
    """GET /movies?limit=100 — Returns the video catalogue for browsing."""
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 1000))
    except ValueError:
        limit = 100
    catalogue = [
        {"video_id": vid, **meta}
        for vid, meta in list(video_metadata.items())[:limit]
    ]
    return jsonify({"count": len(catalogue), "movies": catalogue})


@app.errorhandler(404)
def not_found(_): return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(_): return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(debug=False, port=5000)