from sklearn.preprocessing import LabelEncoder
from src.banking_analytics import load_file
import pandas as pd


df = load_file.load_data()

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
    df = pd.cut(df['age'], bins = [0,20,50,40,60,100], labels = ["Young", "Adult", "Middle_age", "senior"])
    return df

def encode_categorical(df):
    df = df.copy()
    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(["object"]).columns
    for column in categorical_columns:
        df[column] = encoder.fit_transform(df[column])

    return df

def save_data(df, file_path):
    df.to_csv(file_path, index = False)
    print(f"File saved successfully {file_path}")

def main():

    output = "/Users/skhansar19/PycharmProjects/bank_customer_analysis/data/bank_final.csv"
    df = handle_missing_values(df)
    df = creat_new_feature(df)
    df = encode_categorical(df)
    save_data(df, output)

if __name__ == "__main__":
    main()




