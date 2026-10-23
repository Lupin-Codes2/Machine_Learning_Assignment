import csv
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / "Top 1000 games of all time.csv"
MODEL_DIR = ROOT_DIR / "models"
REGRESSION_FEATURES = ["ranking", "year"]
CLASSIFICATION_FEATURES = ["ranking", "year", "votes"]
MIN_GENRE_SAMPLES = 15


def load_games() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    regression_rows = []
    classification_rows = []
    genres = []
    with DATASET_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                ranking = float(row["Ranking"].replace(",", "").strip())
                votes = int(row["Votes"].replace(",", "").strip())
                year_match = re.search(r"\d{4}", row["Year"])
                primary_genre = row["Genre"].split(",", maxsplit=1)[0].strip()
                if not year_match or not primary_genre:
                    continue
                year = int(year_match.group())
            except (KeyError, ValueError, AttributeError):
                continue

            regression_rows.append([ranking, year])
            classification_rows.append([ranking, year, votes])
            genres.append(primary_genre)

    if len(regression_rows) < 50:
        raise ValueError(
            f"Expected at least 50 usable game records in {DATASET_PATH}; "
            f"found {len(regression_rows)}."
        )

    return (
        np.asarray(regression_rows, dtype=float),
        np.asarray(classification_rows, dtype=float),
        np.asarray(genres, dtype=str),
    )


def train_regression_model(
    features: np.ndarray, votes: np.ndarray
) -> tuple[dict, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        features, votes, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    score = float(r2_score(y_test, model.predict(x_test)))
    return {
        "model": model,
        "feature_names": REGRESSION_FEATURES,
        "metric": {"name": "R²", "value": score},
        "target_name": "votes",
    }, score


def train_classification_model(
    features: np.ndarray, genres: np.ndarray
) -> tuple[dict, float]:
    genre_counts = Counter(genres)
    labels = np.asarray(
        [
            genre if genre_counts[genre] >= MIN_GENRE_SAMPLES else "Other"
            for genre in genres
        ],
        dtype=str,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    model.fit(x_train, y_train)
    score = float(accuracy_score(y_test, model.predict(x_test)))
    return {
        "model": model,
        "feature_names": CLASSIFICATION_FEATURES,
        "class_names": model.classes_.tolist(),
        "metric": {"name": "Accuracy", "value": score},
        "min_genre_samples": MIN_GENRE_SAMPLES,
    }, score


def main() -> None:
    regression_features, classification_features, genres = load_games()
    votes = classification_features[:, 2]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    regression_bundle, r2 = train_regression_model(regression_features, votes)
    classification_bundle, accuracy = train_classification_model(
        classification_features, genres
    )
    joblib.dump(regression_bundle, MODEL_DIR / "regression_model.joblib")
    joblib.dump(classification_bundle, MODEL_DIR / "classification_model.joblib")
    print(f"Usable game records: {len(votes)}")
    print(f"Vote regression test R²: {r2:.4f}")
    print(f"Primary genre classification test accuracy: {accuracy:.4f}")
    print(f"Saved both models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
