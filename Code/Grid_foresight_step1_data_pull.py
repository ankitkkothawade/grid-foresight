
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

# --- Where your manually-downloaded data lives --------------------------
# Anchored to this script's own folder, not the current working directory —
# so it works the same whether you run it from your IDE, terminal, or
# double-click it. Assumes a layout like:
#   Grid Forecasting/
#     Code/    <- this script lives here
#     data/    <- your CSVs live here (sibling of Code, not nested inside it)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"/ "raw"  # put one or more NESO CSVs directly in here, OR:
SINGLE_FILE = None  # e.g. SCRIPT_DIR.parent / "data" / "my_combined_demand_data.csv"

# The name step 1 saves its own combined output as — excluded from the
# "raw input" glob below so re-running this script doesn't feed its own
# previous output back in as if it were another year's raw file.
OUTPUT_FILENAME = "demand_combined.csv"


def load_local_csvs(data_dir: Path, single_file: Optional[Path]) -> pd.DataFrame:
    if single_file is not None:
        print(f"Loading {single_file}...")
        return pd.read_csv(single_file)

    csv_files = sorted(
        f for f in data_dir.glob("*.csv") if f.name != OUTPUT_FILENAME
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir.resolve()}. "
            "Download the year files from the NESO Data Portal and place "
            "them there, or set SINGLE_FILE to point at one combined CSV."
        )

    frames = []
    for f in csv_files:
        print(f"Loading {f.name}...")
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_local_csvs(DATA_DIR, SINGLE_FILE)

    print("\n--- First look ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    # NESO's half-hourly files use SETTLEMENT_DATE + SETTLEMENT_PERIOD
    # (1-48, occasionally 46/50 on clock-change days), plus ND (National
    # Demand, MW) and TSD (Transmission System Demand, MW) — confirmed from
    # NESO's own column documentation. I have NOT independently re-verified
    # every other column name today, so print df.columns above and adjust
    # the two lines below if anything doesn't match what you see.
    date_col = "SETTLEMENT_DATE"
    period_col = "SETTLEMENT_PERIOD"

    if date_col in df.columns and period_col in df.columns:
        print(f"\nSample raw {date_col} values:", df[date_col].head(3).tolist())

        # Don't assume a single date format across years of CSVs — NESO's
        # own format has drifted over time, and guessing wrong (as I did
        # with dayfirst=True) silently misparses rows instead of failing
        # loudly. Try strict ISO first (fast, and correct if it works),
        # then fall back to per-row inference if that fails.
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="ISO8601")
        except ValueError:
            print(f"ISO8601 parse failed, falling back to mixed-format inference...")
            df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True)

        df["datetime"] = df[date_col] + pd.to_timedelta(
            (df[period_col] - 1) * 30, unit="min"
        )
        df = df.sort_values("datetime").reset_index(drop=True)

        duplicate_count = df.duplicated(subset=["datetime"]).sum()
        if duplicate_count > 0:
            print(
                f"\nFound {duplicate_count} duplicate datetime rows out of "
                f"{len(df)} total — likely from two of your downloaded CSVs "
                "covering overlapping dates (e.g. a 'current year update' "
                "file that overlaps an archived year file). Dropping "
                "duplicates, keeping the last occurrence. Worth checking "
                "your data/ folder for files with overlapping date ranges "
                "if you want to understand exactly where these came from."
            )
            df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    else:
        print(
            "\nWARNING: expected columns not found. Check the printed "
            "column list above and update date_col/period_col to match."
        )

    out_path = DATA_DIR / "demand_combined.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved combined dataset to {out_path}")

    if "ND" in df.columns and "datetime" in df.columns:
        plt.figure(figsize=(14, 5))
        plt.plot(df["datetime"], df["ND"], linewidth=0.5)
        plt.title("GB National Demand (ND), half-hourly")
        plt.xlabel("Date")
        plt.ylabel("Demand (MW)")
        plt.tight_layout()
        plot_path = SCRIPT_DIR / "first_look_demand.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")
    else:
        print("Couldn't find 'ND' column for plotting — check df.columns above.")


if __name__ == "__main__":
    main()