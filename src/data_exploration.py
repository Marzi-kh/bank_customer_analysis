import pandas as pd
from src import load_file


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

    if "deposit" in df.columns:
        print("\n" + "=" * 50)
        print("TARGET DISTRIBUTION (deposit)")
        print("=" * 50)
        print(df["deposit"].value_counts())

        print("\nPercentages:")
        print(df["deposit"].value_counts(normalize=True) * 100)




def save_clean_data(df, file_path):
    """ Save cleaned data to csv file"""
    df.to_csv(file_path, index=False)
    print(f"\nData saved to {file_path}")


def main():
    df = load_file.load_data()
    explore_df(df)

if __name__ == "__main__":
    main()