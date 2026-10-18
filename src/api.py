from pathlib import Path
import logging
from collections import Counter
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


# Configuring API request logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Defining the project paths
project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "models" / "heart_disease_pipeline.joblib"

# Loading the trained preprocessing and modelling pipeline
model = joblib.load(model_path)

# Initializing thread-safe request and prediction counters
metrics = Counter()
metrics_lock = Lock()


# Defining the expected patient input fields
class PatientData(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=3)
    thal: float = Field(ge=3, le=7)


# Initializing the FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predict heart disease labels using a trained ML pipeline. "
        "For educational use only; not a medical diagnosis."
    ),
    version="1.1.0",
)


# Recording incoming HTTP requests
@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception:
        with metrics_lock:
            metrics["http_requests_total"] += 1
            metrics["http_requests_5xx_total"] += 1
        raise

    with metrics_lock:
        metrics["http_requests_total"] += 1

        if response.status_code >= 500:
            metrics["http_requests_5xx_total"] += 1
        elif response.status_code >= 400:
            metrics["http_requests_4xx_total"] += 1

    return response


@app.get("/health")
def health_check():
    # Reporting the API health status
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    # Returning a snapshot of API request and prediction counters
    with metrics_lock:
        return {
            "http_requests_total": metrics["http_requests_total"],
            "http_requests_4xx_total": metrics["http_requests_4xx_total"],
            "http_requests_5xx_total": metrics["http_requests_5xx_total"],
            "predictions_total": metrics["predictions_total"],
            "predictions_positive_total": metrics["predictions_positive_total"],
            "predictions_negative_total": metrics["predictions_negative_total"],
            "prediction_errors_total": metrics["prediction_errors_total"],
        }


@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    # Converting the validated patient data into a dataframe
    patient_df = pd.DataFrame([patient.model_dump()])

    # Generating the prediction and positive-class probability
    try:
        prediction = int(model.predict(patient_df)[0])
        probabilities = model.predict_proba(patient_df)[0]
        confidence = float(probabilities[prediction])

        # Recording prediction outcomes
        with metrics_lock:
            metrics["predictions_total"] += 1

            if prediction == 1:
                metrics["predictions_positive_total"] += 1
            else:
                metrics["predictions_negative_total"] += 1

        # Recording the prediction request
        logger.info(
            "Prediction request processed: prediction=%s, confidence=%.4f",
            prediction,
            confidence,
        )

        return {
            "prediction": prediction,
            "label": (
                "Heart disease present"
                if prediction == 1
                else "Heart disease absent"
            ),
            "confidence": round(confidence, 4),
            "disclaimer": "For educational use only; not a medical diagnosis.",
        }

    except Exception:
        # Recording and logging unexpected prediction errors
        with metrics_lock:
            metrics["prediction_errors_total"] += 1

        logger.exception("Failed to process prediction request")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        )
