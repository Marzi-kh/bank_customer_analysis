from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.load_file import load_data


def train_baseline_model(top_n: int = 100):
    df = load_data()



    df["pdays_was_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)


    X = df.drop(columns=["deposit", "duration"])  # remove duration to avoid target leakage
    y = df["deposit"].map({"yes": 1, "no": 0})


    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:

        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_prob))

    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = y_pred
    results["probability"] = y_prob

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_sorted = results.sort_values(by="probability", ascending=False)
    top_customers = results_sorted.head(top_n)

    top_out_path = output_dir / "top_customers.csv"
    top_customers.to_csv(top_out_path, index=False)
    print(f"\nTop {top_n} customers saved to: {top_out_path}")


    if "age_group" not in results.columns and "age" in results.columns:
        results["age_group"] = pd.cut(
            results["age"],
            bins=[0, 20, 50, 60, 100],
            labels=["Young", "Adult", "Middle_age", "Senior"],
            include_lowest=True,
        )

    def make_segment_summary(df_: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if group_col not in df_.columns:
            return pd.DataFrame()

        summary = (
            df_.groupby(group_col, observed=False)
            .agg(
                customers=("actual", "size"),
                actual_yes_rate=("actual", "mean"),
                avg_predicted_probability=("probability", "mean"),
            )
            .sort_values("avg_predicted_probability", ascending=False)
            .reset_index()
        )
        return summary

    summaries = []

    job_summary = make_segment_summary(results, "job")
    if not job_summary.empty:
        summaries.append(job_summary.assign(segment_type="job").rename(columns={"job": "segment"}))

    age_summary = make_segment_summary(results, "age_group")
    if not age_summary.empty:
        summaries.append(age_summary.assign(segment_type="age_group").rename(columns={"age_group": "segment"}))

    marital_summary = make_segment_summary(results, "marital")
    if not marital_summary.empty:
        summaries.append(marital_summary.assign(segment_type="marital").rename(columns={"marital": "segment"}))

    if summaries:
        segment_summary = pd.concat(summaries, ignore_index=True)
        segment_out_path = output_dir / "segment_summary.csv"
        segment_summary.to_csv(segment_out_path, index=False)
        print(f"Segment summary saved to: {segment_out_path}")
    else:
        print("Segment summary not saved (no segment columns found).")

    return model


def main():
    train_baseline_model(top_n=100)


if __name__ == "__main__":
    main()