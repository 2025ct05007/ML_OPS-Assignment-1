# Importing the testing utilities
from fastapi.testclient import TestClient

# Importing the FastAPI application
from src.api import app

# Creating a test client for the API
client = TestClient(app)


# Testing the health endpoint
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# Testing a valid prediction request
def test_predict_endpoint():
    payload = {
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
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in [
        "Heart disease absent",
        "Heart disease present",
    ]
    assert 0 <= data["confidence"] <= 1
    assert "disclaimer" in data


# Testing validation for an invalid request
def test_predict_validation_error():
    payload = {
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
    }

    # Providing an invalid age to trigger request validation
    payload["age"] = 150

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
