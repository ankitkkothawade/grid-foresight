
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NESO_FORECAST_FILE = SCRIPT_DIR.parent / "data" / "raw" / "CHANGE_ME.csv"  # match step4a
DEMAND_FILE = SCRIPT_DIR.parent / "data" / "raw" / "demand_combined.csv"

# The exact same four fold windows from step 3's output -- hardcoded here
# so this comparison is guaranteed apples-to-apples with those MAPE numbers,
# not just "close enough" windows.
FOLD_WINDOWS = [
    ("2026-06-12", "2026-06-26"),
    ("2026-06-26", "2026-07-10"),
    ("2026-07-10", "2026-07-24"),
    ("2026-07-24", "2026-08-07"),
]

# XGBoost's MAPE per fold, from step 3's actual output -- not recomputed
# here, just carried over for the side-by-side comparison.
XGBOOST_MAPE_BY_FOLD = [5.08, 7.08, 5.47, 5.83]


def load_neso_forecast() -> pd.DataFrame:
    df = pd.read_csv(NESO_FORECAST_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["datetime"] = df["Date"] + pd.to_timedelta((df["Settlement_Period"] - 1) * 30, unit="min")
    return df


def sanity_check_against_own_data(neso: pd.DataFrame):
    """Cross-check NESO's own Demand_Outturn against our ND column for a
    handful of overlapping rows -- if these don't roughly match, something
    is misaligned before we trust the APE comparison at all."""
    try:
        demand = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"])
        merged = neso.merge(demand[["datetime", "ND"]], on="datetime", how="inner")
        if len(merged) == 0:
            print("WARNING: no overlapping timestamps found for sanity check -- "
                  "check that both files' datetime construction lines up.")
            return
        diff_pct = ((merged["Demand_Outturn"] - merged["ND"]).abs() / merged["ND"]).mean() * 100
        print(f"Sanity check: NESO's own Demand_Outturn vs. our ND column "
              f"differ by {diff_pct:.3f}% on average across {len(merged)} overlapping rows.")
        print("(Should be at or near 0% -- these are both supposed to be the same "
              "real outturn demand figure, just from two different NESO datasets.)\n")
    except FileNotFoundError:
        print("(Skipping sanity check -- demand_combined.csv not found at expected path.)\n")


def main():
    neso = load_neso_forecast()

    print(f"APE column range: min={neso['APE'].min():.4f}, max={neso['APE'].max():.4f}, "
          f"mean={neso['APE'].mean():.4f}")
    print("(Checking scale before trusting it as a percentage -- if these values "
          "look like small decimals under 1, APE is stored as a fraction and needs "
          "x100; if they're already in normal percentage range, it's fine as-is.)\n")

    sanity_check_against_own_data(neso)

    print("=== NESO's own day-ahead forecast vs. our XGBoost, same windows ===\n")
    neso_apes = []
    for i, ((start, end), xgb_mape) in enumerate(zip(FOLD_WINDOWS, XGBOOST_MAPE_BY_FOLD), start=1):
        window = neso[(neso["datetime"] >= start) & (neso["datetime"] < end)]
        neso_mape = window["APE"].mean()
        neso_apes.append(neso_mape)
        print(f"Fold {i} ({start} to {end}): "
              f"NESO {neso_mape:.2f}% MAPE  vs.  our XGBoost {xgb_mape:.2f}% MAPE")

    print(f"\nOverall: NESO {sum(neso_apes) / len(neso_apes):.2f}% MAPE  "
          f"vs.  our XGBoost {sum(XGBOOST_MAPE_BY_FOLD) / len(XGBOOST_MAPE_BY_FOLD):.2f}% MAPE")


if __name__ == "__main__":
    main()
