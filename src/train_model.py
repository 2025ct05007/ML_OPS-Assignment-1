import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def train_models():
    # Defining the project paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "heart_disease.csv"
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Loading the prepared dataset
    df = pd.read_csv(data_path)

    # Creating the binary target
    y = (df["target"] > 0).astype(int)

    # Separating the input features from the target
    X = df.drop(columns=["target"])

    # Defining categorical and numerical features
    categorical_features = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal",
    ]
    numerical_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]

    # Creating the numerical preprocessing pipeline
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Creating the categorical preprocessing pipeline
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combining numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Splitting the dataset while preserving target proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Defining the candidate models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        ),
    }

    # Configuring the MLflow experiment
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(
        (project_root / "mlruns").resolve().as_uri()
    )
    mlflow.set_experiment("Heart Disease Classification")

    # Defining stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    best_model = None
    best_model_name = None
    best_auc = -1.0

    for model_name, classifier in models.items():
        # Creating the complete preprocessing and modelling pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        with mlflow.start_run(run_name=model_name):
            # Evaluating the model using cross-validation
            cv_results = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "roc_auc": "roc_auc",
                },
                n_jobs=1,
            )

            # Training the pipeline on the complete training set
            pipeline.fit(X_train, y_train)

            # Evaluating predictions on the held-out test set
            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
                "cv_precision_mean": cv_results["test_precision"].mean(),
                "cv_recall_mean": cv_results["test_recall"].mean(),
                "cv_roc_auc_mean": cv_results["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            # Logging the model parameters and evaluation metrics
            mlflow.log_params(classifier.get_params())
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(
                {key: float(value) for key, value in metrics.items()}
            )

            # Logging the trained preprocessing and modelling pipeline
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            # Displaying the evaluation results
            print(f"\n{'=' * 50}")
            print(f"Model: {model_name}")
            print(f"{'=' * 50}")

            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

            # Selecting the best model using cross-validation ROC-AUC
            current_auc = metrics["cv_roc_auc_mean"]

            if current_auc > best_auc:
                best_auc = current_auc
                best_model = pipeline
                best_model_name = model_name

    # Saving the selected model pipeline
    import joblib

    model_path = model_dir / "heart_disease_pipeline.joblib"
    joblib.dump(best_model, model_path)

    # Recording the selected model
    print(f"\nSelected model: {best_model_name}")
    print(f"Cross-validation ROC-AUC: {best_auc:.4f}")
    print(f"Saved model pipeline to: {model_path}")


if __name__ == "__main__":
    train_models()
