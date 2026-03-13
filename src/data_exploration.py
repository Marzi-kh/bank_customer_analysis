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



def eda_quick_numbers(df, target="deposit"):
    d = df.copy()

    # convert target to binary
    d["target_bin"] = d[target].astype(str).str.lower().map({"yes": 1, "no": 0})

    if d["target_bin"].isna().any():
        print("Warning: some target values could not be mapped to binary.")

    print("\n=== BASIC CHECKS ===")
    print("Rows:", len(d))
    print("Duplicates:", d.duplicated().sum())

    print("\nMissing values:")
    print(d.isna().sum())

    print("\n=== TARGET DISTRIBUTION (%) ===")
    print((d["target_bin"].value_counts(normalize=True) * 100).round(2))

    # ---- categorical features
    cat_cols = d.select_dtypes(include=["object", "category"]).columns
    cat_cols = [c for c in cat_cols if c != target]

    print("\n=== CATEGORICAL FEATURES (Subscription Rate %) ===")
    for col in cat_cols:
        print(f"\n-- {col} --")
        summary = d.groupby(col)["target_bin"].agg(
            customers="count",
            subscription_rate="mean"
        )
        summary["subscription_rate"] = (summary["subscription_rate"] * 100).round(2)
        print(summary.sort_values("subscription_rate", ascending=False))

    # ---- numeric features
    num_cols = d.select_dtypes(include="number").columns
    num_cols = [c for c in num_cols if c != "target_bin"]

    print("\n=== NUMERIC FEATURES (Mean by class) ===")
    summary = d.groupby("target_bin")[num_cols].mean().T
    summary.columns = ["No_mean", "Yes_mean"]
    summary["diff"] = summary["Yes_mean"] - summary["No_mean"]

    print(summary.sort_values("diff", ascending=False).round(2))


def main():
    df = load_file.load_data()
    explore_df(df)
    eda_quick_numbers(df)


if __name__ == "__main__":
    main()
