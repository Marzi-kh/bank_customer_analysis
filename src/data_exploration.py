import pandas as pd
from src.banking_analytics import load_file

df = load_file.load_data()
def load_data(file_path):
    """ Loads data from csv file"""
    data = pd.read_csv(file_path)
    return data

def explore_df(df):
    """ Perform basic exploration : head, info"""
    print("First five rows of dataframe:")
    print(df.head(5))

    print("\nSecond five rows of dataframe:")
    print(df.tail(5))

    print("\nDataset information:")
    print(df.info())

    print("\nStatistical summary:")
    print(df.describe())

    print("\n'missing Value per Column:")
    print(df.isnull().sum())

def save_clean_data(df, file_path):
    """ Save cleaned data to csv file"""
    df.to_csv(file_path, index=False)
    print(f"\nData saved to {file_path}")


def main():
    data = load_data("/Users/skhansar19/PycharmProjects/bank_customer_analysis/data/bank.csv")
    explore_df(data)

if __name__ == "__main__":
    main()