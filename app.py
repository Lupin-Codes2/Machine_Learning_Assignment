from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
STATIC_DIR = ROOT_DIR / "static"


def load_model(filename: str) -> dict[str, Any]:
    model_path = MODEL_DIR / filename
    if not model_path.is_file():
        raise RuntimeError(
            f"Model file not found: {model_path}. Run `python train_models.py` first."
        )
    return joblib.load(model_path)


regression_bundle = load_model("regression_model.joblib")
classification_bundle = load_model("classification_model.joblib")

app = FastAPI(title="Full-Stack Machine Learning Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RegressionFeatures(BaseModel):
    ranking: float = Field(ge=1, le=1000)
    year: int = Field(ge=1950, le=2030)


class RegressionRequest(BaseModel):
    features: RegressionFeatures


class ClassificationFeatures(BaseModel):
    ranking: float = Field(ge=1, le=1000)
    year: int = Field(ge=1950, le=2030)
    votes: int = Field(ge=0, le=10_000_000)


class ClassificationRequest(BaseModel):
    features: ClassificationFeatures


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/regression")
def predict_regression(request: RegressionRequest) -> dict[str, int | float | str]:
    features = request.features.model_dump()
    values = np.array(
        [[features[name] for name in regression_bundle["feature_names"]]],
        dtype=float,
    )
    prediction = float(regression_bundle["model"].predict(values)[0])
    return {
        "prediction": max(0, round(prediction)),
        "unit": "votes",
        "r2": round(regression_bundle["metric"]["value"], 4),
    }


@app.post("/predict/classification")
def predict_classification(
    request: ClassificationRequest,
) -> dict[str, int | str | float]:
    features = request.features.model_dump()
    values = np.array(
        [[features[name] for name in classification_bundle["feature_names"]]],
        dtype=float,
    )
    predicted_genre = str(classification_bundle["model"].predict(values)[0])
    class_names = classification_bundle["class_names"]
    probabilities = classification_bundle["model"].predict_proba(values)[0]
    class_id = class_names.index(predicted_genre)
    return {
        "prediction": predicted_genre,
        "class_id": class_id,
        "confidence": round(float(probabilities[class_id]), 4),
        "accuracy": round(classification_bundle["metric"]["value"], 4),
    }
