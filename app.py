"""Clio — Flask API + static UI (Hugging Face Spaces)."""

import logging
import os
import re
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

from model import (
    USER_ID_RE,
    ClioModel,
    build_model,
    recommend_for_user,
    record_feedback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_ROOT = Path(os.environ.get("STATIC_ROOT", "static"))
app = Flask(__name__, static_folder=None)
CORS(app)

try:
    clio: ClioModel = build_model(logger=logger)
except Exception as exc:
    logger.exception("Model init failed: %s", exc)
    raise SystemExit(1) from exc


def is_valid_user_id(uid: str) -> bool:
    return bool(uid and re.match(USER_ID_RE, uid))


def is_valid_video_id(vid: str) -> bool:
    return bool(vid) and len(vid) <= 64


def parse_n(raw) -> int | None:
    if raw is None:
        return 10
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 50 else None


@app.route("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "model": "ALS (implicit)",
            "num_users": clio.num_users,
            "num_videos": clio.num_videos,
        }
    )


@app.route("/api/recommendations/<user_id>")
def api_recommendations(user_id):
    if not is_valid_user_id(user_id):
        return jsonify(
            {
                "error": "invalid_user_id",
                "message": "user_id must be 1–64 alphanumeric characters.",
            }
        ), 400

    n = parse_n(request.args.get("n"))
    if n is None:
        return jsonify(
            {
                "error": "invalid_n",
                "message": "n must be an integer between 1 and 50.",
            }
        ), 400

    try:
        return jsonify(recommend_for_user(clio, user_id, n))
    except LookupError:
        return jsonify(
            {
                "error": "not_found",
                "message": f"Unknown user_id '{user_id}'",
            }
        ), 404
    except Exception as exc:
        logger.exception("Recommend error for %s: %s", user_id, exc)
        return jsonify(
            {
                "error": "internal_error",
                "message": "The recommendation service encountered an error.",
            }
        ), 500


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id", "")).strip()
    video_id = str(body.get("video_id", "")).strip()
    signal = str(body.get("signal", "")).strip().lower()

    if not is_valid_user_id(user_id):
        return jsonify(
            {"error": "invalid_user_id", "message": "user_id is required."}
        ), 400
    if not is_valid_video_id(video_id):
        return jsonify(
            {"error": "invalid_video_id", "message": "video_id is required."}
        ), 400
    if signal not in ("up", "down"):
        return jsonify(
            {"error": "invalid_signal", "message": "signal must be 'up' or 'down'."}
        ), 400

    return jsonify(record_feedback(clio, user_id, video_id, signal))


@app.route("/api/movies")
def api_movies():
    try:
        limit = max(1, min(int(request.args.get("limit", 100)), 1000))
    except ValueError:
        limit = 100
    catalogue = [
        {"video_id": vid, **meta}
        for vid, meta in list(clio.video_metadata.items())[:limit]
    ]
    return jsonify({"count": len(catalogue), "movies": catalogue})


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        abort(404)

    if STATIC_ROOT.is_dir():
        if path:
            target = STATIC_ROOT / path
            if target.is_file():
                return send_from_directory(STATIC_ROOT, path)
        if (STATIC_ROOT / "index.html").is_file():
            return send_from_directory(STATIC_ROOT, "index.html")

    return jsonify(
        {
            "message": "Clio API is running. Build the frontend into STATIC_ROOT to serve the UI.",
            "health": "/api/health",
        }
    )


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "not_found", "message": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "method_not_allowed", "message": "Method not allowed."}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
