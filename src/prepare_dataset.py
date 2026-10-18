
from pathlib import Path
import pandas as pd

def prepare_heart_disease_data():
    # Defining the project and data paths
    project_root = Path(__file__).resolve().parents[1]
    raw_data_path = project_root / "data" / "raw" / "processed.cleveland.data"
    processed_data_dir = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Defining the feature names documented by the UCI repository
    column_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    # Loading the raw Cleveland dataset
    df = pd.read_csv(
        raw_data_path,
        header=None,
        names=column_names,
        na_values="?",
    )

    # Displaying the dataset dimensions and missing values
    print("Dataset loaded successfully.")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values by column:")
    print(df.isnull().sum())

    # Saving the labelled dataset as a CSV file
    output_path = processed_data_dir / "heart_disease.csv"
    df.to_csv(output_path, index=False)

    print(f"\nProcessed dataset saved to: {output_path}")


if __name__ == "__main__":
    prepare_heart_disease_data()
