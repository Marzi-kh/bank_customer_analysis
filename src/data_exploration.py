import pandas as pd
from src import load_file


def explore_df(df):
    """ Perform basic exploration : head, info"""
    print("First five rows of dataframe:")
    print(df.head(5))

    print("\nLast five rows of dataframe:")
    print(df.tail(5))

    print("\nDataset information:")
    df.info()

    print("\nStatistical summary:")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\n" + "=" * 5)
    print("DUPLICATE ROWS")
    print("=" * 5)
    print(df.duplicated().sum())

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    print("\n" + "=" * 50)
    print("NUMERICAL COLUMNS")
    print("=" * 50)
    print(list(num_cols))

    print("\n" + "=" * 50)
    print("CATEGORICAL COLUMNS")
    print("=" * 50)
    print(list(cat_cols))

    print("\n" + "=" * 50)
    print("UNIQUE VALUES (Categorical)")
    print("=" * 50)
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")

    if "deposit" in df.columns:
        print("\n" + "=" * 50)
        print("TARGET DISTRIBUTION (deposit)")
        print("=" * 50)
        print(df["deposit"].value_counts())

        print("\nPercentages:")
        print(df["deposit"].value_counts(normalize=True) * 100)


def main():
    df = load_file.load_data()
    explore_df(df)


if __name__ == "__main__":
    main()
