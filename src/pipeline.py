from src.load_file import load_data
from src.data_cleaning import handle_missing_values, create_new_feature
from src.train_baseline import train_baseline_model

def main():
    df = load_data()
    df = handle_missing_values(df)
    df = create_new_feature(df)

    train_baseline_model(top_n=100)

if __name__ == "__main__":
    main()