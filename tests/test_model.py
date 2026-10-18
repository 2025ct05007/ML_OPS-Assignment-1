import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "heart_disease_pipeline.joblib"

@pytest.fixture(scope="module")
def model():
    # Loading the saved preprocessing and prediction pipeline
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_patient():
    # Creating a valid sample patient record
    return pd.DataFrame([{
        "age": 55,
        "sex": 1,
        "cp": 3,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }])


# Testing that the trained pipeline produces a valid binary prediction
def test_model_prediction_is_binary(model, sample_patient):
    prediction = model.predict(sample_patient)

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)


# Testing that the trained pipeline produces a valid probability
def test_model_prediction_probability(model, sample_patient):
    probabilities = model.predict_proba(sample_patient)

    assert probabilities.shape == (1, 2)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)


# Testing that the pipeline can handle missing feature values
def test_model_handles_missing_values(model, sample_patient):
    sample_patient.loc[0, "chol"] = np.nan

    prediction = model.predict(sample_patient)

    assert len(prediction) == 1
    assert int(prediction[0]) in (0, 1)


# Testing that the input contains the expected model features
def test_model_rejects_missing_feature(model, sample_patient):
    incomplete_patient = sample_patient.drop(columns=["age"])

    with pytest.raises((ValueError, KeyError)):
        model.predict(incomplete_patient)
