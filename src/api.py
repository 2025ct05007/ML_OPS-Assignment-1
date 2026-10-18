from pathlib import Path
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
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
    description="Predicting heart disease labels using a trained ML pipeline.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    # Reporting the API health status
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_heart_disease(patient: PatientData):
    # Converting the validated patient data into a dataframe
    patient_df = pd.DataFrame([patient.model_dump()])

    # Generating the prediction and positive-class probability
    try:
        prediction = int(model.predict(patient_df)[0])
        probabilities = model.predict_proba(patient_df)[0]
        confidence = float(probabilities[prediction])

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
            "disclaimer": (
                "For educational use only; not a medical diagnosis."
            ),
        }

    except Exception:
        # Logging unexpected prediction errors without exposing internals
        logger.exception("Failed to process prediction request")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        )
