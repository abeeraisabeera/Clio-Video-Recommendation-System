import io
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
OUT_INTERACTIONS = Path("clio_interactions.csv")
OUT_MOVIES = Path("movies.csv")

GENRE_COLS = [
    "unknown","Action","Adventure","Animation","Children's","Comedy",
    "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",
]


def download_movielens() -> bytes:
    print(f"Downloading MovieLens 100K from {MOVIELENS_URL} …")
    with urllib.request.urlopen(MOVIELENS_URL) as resp:
        data = resp.read()
    print(f"  Downloaded {len(data)/1024:.0f} KB")
    return data


def parse_ratings(z: zipfile.ZipFile) -> pd.DataFrame:
    raw = z.read("ml-100k/u.data").decode()

    df = pd.read_csv(
        io.StringIO(raw),
        sep="\t",
        names=["user_id", "video_id", "rating", "timestamp"],
    )

    # =========================================================
    # FIX 1: STRONGER SIGNAL TRANSFORMATION (CRITICAL)
    # =========================================================
    # instead of weak normalization, we create real confidence
    df["weight"] = (df["rating"] - 2.5).clip(lower=0)

    # amplify signal separation
    df["weight"] = df["weight"].apply(lambda x: 1 if x == 0 else (3 if x == 1 else (6 if x == 2 else 10)))

    # implicit learning expects repeated strength
    df["weight"] = df["weight"].astype(float)

    df["user_id"] = "user_" + df["user_id"].astype(str)
    df["video_id"] = "movie_" + df["video_id"].astype(str)

    return df[["user_id", "video_id", "weight"]]


def parse_movies(z: zipfile.ZipFile) -> pd.DataFrame:
    raw = z.read("ml-100k/u.item").decode(errors="replace")

    cols = ["video_id","title","release_date","video_release_date","imdb_url"] + GENRE_COLS
    df = pd.read_csv(io.StringIO(raw), sep="|", names=cols, encoding="latin-1")

    df["video_id"] = "movie_" + df["video_id"].astype(str)

    def primary_genre(row):
        for g in GENRE_COLS:
            if row.get(g, 0) == 1:
                return g
        return "General"

    df["category"] = df.apply(primary_genre, axis=1)

    df["thumbnail_url"] = df["video_id"].apply(
        lambda vid: f"https://picsum.photos/seed/{vid}/320/180"
    )

    import hashlib
    def stable_duration(vid):
        h = int(hashlib.md5(vid.encode()).hexdigest(), 16)
        return 80*60 + (h % (70*60))

    df["duration_seconds"] = df["video_id"].apply(stable_duration)

    return df[["video_id","title","category","thumbnail_url","duration_seconds"]]


def main():
    raw = download_movielens()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        ratings = parse_ratings(z)
        movies = parse_movies(z)

    # =========================================================
    # FIX 2: REMOVE DUPLICATE DILUTION (IMPORTANT FOR ALS)
    # =========================================================
    ratings = ratings.groupby(
        ["user_id", "video_id"],
        as_index=False
    )["weight"].sum()

    interactions = ratings.merge(
        movies[["video_id","title","category","thumbnail_url","duration_seconds"]],
        on="video_id",
        how="left",
    )

    interactions.to_csv(OUT_INTERACTIONS, index=False)
    movies.to_csv(OUT_MOVIES, index=False)

    print(f"\n✓ {OUT_INTERACTIONS} — {len(interactions):,} rows")
    print(f"✓ {OUT_MOVIES} — {len(movies):,} movies")
    print("\nReady: python app.py")


if __name__ == "__main__":
    main()