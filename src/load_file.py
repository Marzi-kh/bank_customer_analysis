from pathlib import Path
import pandas as pd

def load_data(path=None):
    if path:
        file_path = Path(path)
    else:
        base_dir = Path(__file__).resolve().parents[1]
        file_path = base_dir / "data" / "bank.csv"

    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)


def main():
    df = load_data()
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:", df.shape)


if __name__ == "__main__":
    main()
