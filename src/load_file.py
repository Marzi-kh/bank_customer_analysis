from pathlib import Path
import pandas as pd

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / "data" / "bank.csv"
    return pd.read_csv(file_path)

