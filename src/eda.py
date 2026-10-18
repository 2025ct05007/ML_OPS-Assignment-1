
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def perform_eda():
    # Defining the project paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "heart_disease.csv"
    output_dir = project_root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loading the prepared dataset
    df = pd.read_csv(data_path)

    # Converting the target into binary labels for analysis
    df["target_binary"] = (df["target"] > 0).astype(int)

    # Setting a consistent visualization style
    sns.set_theme(style="whitegrid", context="notebook")

    # Displaying dataset dimensions and summary statistics
    print("Dataset dimensions:", df.shape)
    print("\nSummary statistics:")
    print(df.describe().round(2))

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nBinary target distribution:")
    print(df["target_binary"].value_counts().sort_index())

    # Plotting feature distributions
    df.drop(columns=["target", "target_binary"]).hist(
        figsize=(16, 12),
        bins=20,
        edgecolor="black",
        color="steelblue",
    )
    plt.suptitle("Heart Disease Dataset - Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=300)
    plt.close()

    # Plotting the binary target class balance
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target_binary", color="steelblue")
    plt.title("Heart Disease - Target Class Balance")
    plt.xlabel("Heart Disease (0 = Absent, 1 = Present)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=300)
    plt.close()

    # Plotting the correlation heatmap
    correlation_data = df.drop(columns=["target"])
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        correlation_data.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
    )
    plt.title("Heart Disease Dataset - Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Saving the summary statistics
    df.describe().round(2).to_csv(
        project_root / "reports" / "summary_statistics.csv"
    )

    print(f"\nEDA figures saved to: {output_dir}")


if __name__ == "__main__":
    perform_eda()
