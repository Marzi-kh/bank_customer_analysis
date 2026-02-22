from pathlib import Path
from src.load_file import load_data

import pandas as pd



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def create_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 20, 50, 60, 100],
        labels=["Young", "Adult", "Middle_age", "Senior"],
        right=True,
        include_lowest=True,
    )
    return df


def save_data(df: pd.DataFrame, file_name: str = "bank_clean.csv") -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"File saved successfully at {output_path}")


def main():
    df = load_data()
    df = handle_missing_values(df)
    df = create_new_feature(df)
    save_data(df, "bank_clean.csv")


if __name__ == "__main__":
    main()
