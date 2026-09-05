
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NESO_FORECAST_FILE = SCRIPT_DIR.parent / "data" / "raw" / "CHANGE_ME.csv"


def main():
    df = pd.read_csv(NESO_FORECAST_FILE)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()