MLOps Assignment 1 - Heart Disease Classification

1. Project Overview

This project implements an end-to-end MLOps pipeline for binary heart disease classification using the UCI Heart Disease dataset. It covers data preparation, exploratory data analysis, model training and evaluation, MLflow experiment tracking, automated API testing, continuous integration, Docker containerization, Kubernetes deployment, and application logging.

NOTE: This project is intended as an educational project and is not a medical diagnostic tool.


2. Architecture

- Data preparation: Load and clean the UCI Cleveland processed dataset.
- Exploratory data analysis: Generate feature distributions, class balance, correlation visualizations, and summary statistics.
- Model training: Compare Logistic Regression and Random Forest using stratified cross-validation and hold-out evaluation.
- Experiment tracking: Log parameters, metrics, and model artifacts with MLflow.
- API: Serve predictions through FastAPI.
- Testing and CI: Run automated API tests using pytest through GitHub Actions.
- Containerization: Package the API and trained model using Docker.
- Deployment: Deploy the containerized API to a local Kubernetes cluster using Kubernetes Deployment and Service manifests.
- Logging: Inspect API health-check and prediction requests through Kubernetes pod logs.


3. Dataset

Source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

The processed Cleveland dataset contains 303 records and 14 columns, including the target. The original target values are converted into a binary label: 0 indicates absence of heart disease and values greater than 0 indicate presence.

Missing values in the ca and thal features are handled through preprocessing pipelines.


4. Models and Evaluation

Two classifiers are evaluated:
- Logistic Regression
- Random Forest

The Logistic Regression pipeline was selected using mean cross-validation ROC-AUC.

| Metric | Logistic Regression — Hold-out Test |
|---|---:|
| Accuracy | 0.8689 |
| Precision | 0.8125 |
| Recall | 0.9286 |
| ROC-AUC | 0.9665 |

These results are based on a small hold-out test set and should not be interpreted as evidence of clinical performance.


5. Run Locally

Create and activate a Python virtual environment, then install dependencies:
- py -3.12 -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install -r requirements.txt

Prepare the dataset and generate EDA outputs:
- python src/prepare_dataset.py
- python src/eda.py

Train and evaluate the models:
- python src/train_model.py

Run the API:
- python -m uvicorn src.api:app --reload
- Open http://127.0.0.1:8000/docs to explore the API.


6. Automated Tests and CI

Run the API tests locally:
- python -m pytest tests/ -v

The GitHub Actions workflow installs the project dependencies and runs the tests on pushes and pull requests.


7. Docker

Build the image:
- docker build -t heart-disease-api:v1 .

Run it locally:
- docker run --rm -p 8000:8000 heart-disease-api:v1
- Check http://127.0.0.1:8000/health.
- Published image: matty16/heart-disease-api:v1


8. Kubernetes Deployment

The Kubernetes manifests are located in k8s/.

Apply them:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods
kubectl get services

For local access, run:
- kubectl port-forward service/heart-disease-api 8001:8000
- Then open http://127.0.0.1:8001/health or http://127.0.0.1:8001/docs.


9. Logging and Evidence

- API request logs can be inspected using: kubectl logs deployment/heart-disease-api --tail=50
- Screenshots of API predictions, health checks, CI execution, Kubernetes deployment, and logs are stored under reports/Screenshots/.


10. Limitations

- The dataset is small and intended for learning and experimentation.
- Model metrics do not establish medical reliability.
- Kubernetes is deployed locally using Docker Desktop, not a public cloud cluster.
- Logs demonstrate request-level activity; a full metrics dashboard and automated alerting are not implemented.


11. Conclusion

This project demonstrates a reproducible workflow spanning data preparation, model experimentation, API testing, CI, Docker packaging, Kubernetes deployment, and operational logging. The repository contains the implementation and evidence required to review the pipeline.