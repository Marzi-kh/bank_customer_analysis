import pandas as pd

def load_data():
    """load cleaned dataset"""
    file_path = "/Users/skhansar19/PycharmProjects/bank_customer_analysis/data/bank.csv"
    df = pd.read_csv(file_path)
    return df

df = load_data()