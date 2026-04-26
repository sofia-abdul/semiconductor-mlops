from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.config import MODEL_PATH, PREDICTION_LOG_PATH


app = FastAPI(
    title="SECOM Yield Prediction API",
    description="API for predicting semiconductor manufacturing pass/fail outcomes.",
    version="1.0.0",
)

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
threshold = model_bundle["threshold"]
target_column = model_bundle["target_column"]

expected_feature_count = getattr(model, "n_features_in_", None)


class PredictionInput(BaseModel):
    features: list[float]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
        "threshold": round(float(threshold), 4),
        "expected_feature_count": expected_feature_count,
    }


@app.post("/predict")
def predict(input_data: PredictionInput):
    if expected_feature_count is not None and len(input_data.features) != expected_feature_count:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {expected_feature_count} features, "
                f"but received {len(input_data.features)}."
            ),
        )

    feature_df = pd.DataFrame([input_data.features])

    if hasattr(model, "predict_proba"):
        failure_probability = float(model.predict_proba(feature_df)[0][1])
    else:
        raw_prediction = int(model.predict(feature_df)[0])
        failure_probability = float(raw_prediction)

    prediction = int(failure_probability >= threshold)

    response = {
        "prediction": prediction,
        "prediction_label": "fail" if prediction == 1 else "pass",
        "failure_probability": round(failure_probability, 4),
        "threshold": round(float(threshold), 4),
        "target_column": target_column,
    }

    log_prediction(response)

    return response


def log_prediction(response: dict) -> None:
    Path(PREDICTION_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction": response["prediction"],
        "prediction_label": response["prediction_label"],
        "failure_probability": response["failure_probability"],
        "threshold": response["threshold"],
    }

    log_df = pd.DataFrame([log_row])

    if Path(PREDICTION_LOG_PATH).exists():
        log_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(PREDICTION_LOG_PATH, index=False)