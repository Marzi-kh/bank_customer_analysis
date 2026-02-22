import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.load_file import load_data

# Better default style
sns.set(style="whitegrid")


def save_plot(name):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / name)
    plt.close()


def plot_age_distribution(df):
    """Plot age distribution"""
    plt.figure()
    sns.histplot(df["age"], bins=30, kde=True)
    plt.title("Age Distribution of Customers")
    plt.xlabel("Age")
    plt.ylabel("Count")
    save_plot("age_distribution.png")


def plot_job_distribution(df):
    """Plot job distribution"""
    plt.figure()
    sns.countplot(x="job", data=df)
    plt.xticks(rotation=45)
    plt.title("Job Distribution of Customers")
    plt.xlabel("Job")
    plt.ylabel("Count")
    save_plot("job_distribution.png")


def plot_correlation_heatmap(df):
    """Plot correlation heatmap"""
    plt.figure()
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()

    sns.heatmap(corr_matrix, annot=True)
    plt.xticks(rotation=45)
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")


def main():
    df = load_data()
    plot_age_distribution(df)
    plot_job_distribution(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    main()