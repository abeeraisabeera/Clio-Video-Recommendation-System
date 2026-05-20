"""
evaluate.py — Offline evaluation for the Clio ALS recommender
"""

import argparse

import numpy as np
import pandas as pd

from model import build_model_from_frame, load_interactions


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    top = recommended[:k]
    return len(set(top) & relevant) / k if k else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevance: dict, k: int) -> float:
    dcg = sum(
        relevance.get(item, 0.0) / np.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(rank + 1) for rank, score in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def load_and_split(csv_path: str, test_ratio: float = 0.2, min_ratings: int = 10):
    df = load_interactions(csv_path)
    counts = df.groupby("user_id")["video_id"].count()
    active = counts[counts >= min_ratings].index
    df = df[df["user_id"].isin(active)].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_rows, test_rows = [], []
    for _, group in df.groupby("user_id"):
        group = group.sample(frac=1, random_state=0)
        split = max(1, int(len(group) * (1 - test_ratio)))
        train_rows.append(group.iloc[:split])
        test_rows.append(group.iloc[split:])

    return pd.concat(train_rows).reset_index(drop=True), pd.concat(test_rows).reset_index(
        drop=True
    )


def evaluate(
    csv_path: str = "clio_interactions.csv",
    k_values: list = (5, 10, 20),
    sample_users: int = 300,
    test_ratio: float = 0.2,
):
    print("Loading data …")
    train, test = load_and_split(csv_path, test_ratio=test_ratio)

    print(f"Training ALS on {len(train):,} interactions …")
    model, matrix, user_enc, video_enc = build_model_from_frame(train)

    known_users = set(user_enc.classes_)
    test_users = [u for u in test["user_id"].unique() if u in known_users]
    if sample_users:
        rng = np.random.default_rng(0)
        test_users = list(
            rng.choice(test_users, size=min(sample_users, len(test_users)), replace=False)
        )

    test_lookup = (
        test[test["user_id"].isin(test_users)]
        .groupby("user_id")
        .apply(lambda g: dict(zip(g["video_id"], g["weight"])))
        .to_dict()
    )

    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    max_k = max(k_values)

    print(f"Evaluating {len(test_users)} users …")
    for user_id in test_users:
        try:
            user_idx = user_enc.transform([user_id])[0]
        except ValueError:
            continue

        relevance = test_lookup.get(user_id, {})
        relevant = set(relevance.keys())
        if not relevant:
            continue

        known_videos = set(video_enc.classes_)
        relevant = relevant & known_videos
        relevance = {v: s for v, s in relevance.items() if v in known_videos}
        if not relevant:
            continue

        ids, _ = model.recommend(
            user_idx,
            matrix[user_idx],
            N=max_k,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        recommended = video_enc.inverse_transform(ids).tolist()

        for k in k_values:
            results[k]["precision"].append(precision_at_k(recommended, relevant, k))
            results[k]["recall"].append(recall_at_k(recommended, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(recommended, relevance, k))

    print("\n" + "=" * 52)
    print("  Clio ALS Recommender — Evaluation Report")
    print(f"  Dataset : {csv_path}")
    print(f"  Split   : {int((1 - test_ratio) * 100)}/{int(test_ratio * 100)} train/test")
    print(f"  Users   : {len(test_users)} sampled")
    print("=" * 52)
    print(f"  {'k':>4}  {'Precision@k':>12}  {'Recall@k':>10}  {'NDCG@k':>8}")
    print("  " + "-" * 40)

    for k in k_values:
        p = np.mean(results[k]["precision"])
        r = np.mean(results[k]["recall"])
        n = np.mean(results[k]["ndcg"])
        print(f"  {k:>4}  {p:>12.4f}  {r:>10.4f}  {n:>8.4f}")

    print("=" * 52)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Clio recommender")
    parser.add_argument("--csv", default="clio_interactions.csv")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--split", type=float, default=0.2)
    args = parser.parse_args()

    k_values = sorted({max(1, args.k // 2), args.k, args.k * 2})
    evaluate(
        csv_path=args.csv,
        k_values=k_values,
        sample_users=args.users,
        test_ratio=args.split,
    )
