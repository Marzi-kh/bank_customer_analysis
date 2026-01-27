from sklearn.preprocessing import LabelEncoder
from src import load_file
from pathlib import Path
import pandas as pd


def handle_missing_values(df):
    #For this dataset, usually 'unknown' is used instead of Nan
    df = df.copy()

    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

def creat_new_feature(df):
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins = [0,20,50,60,100], labels = ["Young", "Adult", "Middle_age", "senior"])
    return df

def encode_categorical(df):
    df = df.copy()
    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(["object"]).columns
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df

def save_data(df, file_name = 'bank_final.csv'):
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"File saved successfully at {output_path}")


def main():
    df = load_file.load_data()
    df = handle_missing_values(df)
    df = creat_new_feature(df)
    df = encode_categorical(df)
    save_data(df, "bank_final.csv")

if __name__ == "__main__":
    main()




