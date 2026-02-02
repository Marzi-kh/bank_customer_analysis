import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.load_file import load_data



def train_baseline_model():

    df = load_data()
    X = df.drop(columns=["deposit"])
    y = df["deposit"].map({"yes":1, "no":0})
    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y
    )

    scaler = StandardScaler()

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        [
            ("num", scaler, num_cols),
            ("cat", encoder, cat_cols)
        ]
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    confusion_matrix(y_test, y_pred)

    classification_report(y_test, y_pred)

    roc_auc_score(y_test, y_prob)

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_prob))



def main():
    train_baseline_model()


if __name__ == "__main__":
    main()









