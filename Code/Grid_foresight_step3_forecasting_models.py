
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Confirmed via a standalone test (clean synthetic matmul, no NaN/Inf in the
# result, warnings fired anyway) that these three specific RuntimeWarnings
# are a known cosmetic bug in NumPy 2.0.2's Accelerate BLAS backend on
# Apple Silicon -- not a sign of real numerical corruption. Silencing them
# here so real output isn't buried in noise; if you ever see a DIFFERENT
# warning, don't assume it's also benign without checking.
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul")
warnings.filterwarnings("ignore", message="overflow encountered in matmul")
warnings.filterwarnings("ignore", message="invalid value encountered in matmul")

SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR.parent / "data" / "raw" / "demand_combined.csv"

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7

N_FOLDS = 4         # how many walk-forward folds to evaluate on
TEST_DAYS = 14      # each fold's test window, in days
FOLD_GAP_DAYS = 14  # folds step backward through history by this many days


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def make_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def make_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1day"] = df["ND"].shift(PERIODS_PER_DAY)
    df["lag_1week"] = df["ND"].shift(PERIODS_PER_WEEK)
    return df


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual = actual.reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    mask = actual != 0
    return float((np.abs(actual[mask] - predicted[mask]) / actual[mask]).mean() * 100)


def get_fold_splits(df: pd.DataFrame):
    """Walk-forward folds: for each fold, train uses ONLY data strictly
    before that fold's test window (no future leakage), and folds step
    backward through history so you get N_FOLDS independent test windows."""
    max_dt = df["datetime"].max()
    folds = []
    for i in range(N_FOLDS):
        test_end = max_dt - pd.Timedelta(days=FOLD_GAP_DAYS * i)
        test_start = test_end - pd.Timedelta(days=TEST_DAYS)
        train = df[df["datetime"] < test_start]
        test = df[(df["datetime"] >= test_start) & (df["datetime"] < test_end)]
        if len(train) > PERIODS_PER_WEEK * 4 and len(test) > 0:
            folds.append((train, test))
    return list(reversed(folds))  # chronological order, for readable output


def evaluate_naive(train: pd.DataFrame, test: pd.DataFrame) -> float:
    combined = pd.concat([train, test]).sort_values("datetime")
    combined["naive"] = combined["ND"].shift(PERIODS_PER_WEEK)
    test_pred = combined.loc[combined["datetime"].isin(test["datetime"]), "naive"]
    return mape(test["ND"], test_pred)


def evaluate_exp_smoothing(train: pd.DataFrame, test: pd.DataFrame) -> float:
    # Holt-Winters with seasonal_periods=48 only needs a recent window to
    # learn daily seasonality (see note below on why 120 days), but the
    # more important fix is damped_trend=True: an UNDAMPED additive trend
    # extrapolates linearly for the entire forecast horizon (672 half-hour
    # steps = 14 days here). Even a small estimated trend compounds over
    # that many steps and can diverge badly from reality — a well-known
    # Holt-Winters failure mode for longer horizons, and a much better fit
    # for what we actually saw (non-monotonic, fold-dependent blowups)
    # than my earlier "optimizer instability" theory, which the numpy
    # diagnostic just ruled out.
    RECENT_WINDOW_DAYS = 120
    cutoff = train["datetime"].max() - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    train_recent = train[train["datetime"] >= cutoff]

    series = train_recent.set_index("datetime")["ND"].asfreq("30min")
    series = series.interpolate()
    model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=PERIODS_PER_DAY,
    ).fit()
    forecast = model.forecast(len(test))
    return mape(test["ND"], forecast)


def evaluate_prophet(train: pd.DataFrame, test: pd.DataFrame) -> float:
    from prophet import Prophet  # imported here so the script still runs
    # end-to-end even if Prophet isn't installed

    prophet_train = train[["datetime", "ND"]].rename(columns={"datetime": "ds", "ND": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    # Reverted from algorithm="Newton" — it had zero effect on the results
    # (identical MAPE, just 5x slower), which confirms the matmul warnings
    # here are the same cosmetic NumPy/Accelerate issue, not a real
    # optimizer failure. Prophet's ~14-17% MAPE (worse than the naive
    # baseline) therefore looks like a genuine finding, not a bug: its
    # smooth Fourier-based seasonality likely doesn't capture the sharp
    # intraday demand transitions as well as XGBoost's lag features do.
    # Worth reporting honestly rather than chasing further — e.g. "Prophet
    # underperformed here, likely because half-hourly demand has sharper
    # transitions than its default seasonality smoothing captures well."
    model.fit(prophet_train)

    future = test[["datetime"]].rename(columns={"datetime": "ds"})
    forecast = model.predict(future)
    return mape(test["ND"], forecast["yhat"])


def evaluate_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> float:
    features = ["hour", "day_of_week", "month", "is_weekend", "lag_1day", "lag_1week"]

    train_fe = make_lag_features(make_calendar_features(train)).dropna(subset=features)

    # Lags for the test window need history from train, so build features on
    # train+test combined, then take just the test-period rows. Using actual
    # (not predicted) lag values here is realistic, not leakage: by the time
    # you forecast "today", you genuinely know yesterday's and last week's
    # actual demand in production.
    combined_fe = make_lag_features(make_calendar_features(pd.concat([train, test])))
    test_fe = combined_fe.tail(len(test))

    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(train_fe[features], train_fe["ND"])
    preds = model.predict(test_fe[features])
    return mape(test["ND"], preds)


def main():
    df = load_data()
    folds = get_fold_splits(df)
    print(f"Running {len(folds)} walk-forward folds "
          f"({TEST_DAYS}-day test windows, {FOLD_GAP_DAYS} days apart)\n")

    results = {"naive": [], "exp_smoothing": [], "xgboost": [], "prophet": []}

    for i, (train, test) in enumerate(folds, start=1):
        print(f"--- Fold {i}: train through {train['datetime'].max().date()}, "
              f"test {test['datetime'].min().date()} to {test['datetime'].max().date()} ---")

        results["naive"].append(evaluate_naive(train, test))
        print(f"  naive:              {results['naive'][-1]:.2f}% MAPE")

        try:
            results["exp_smoothing"].append(evaluate_exp_smoothing(train, test))
            print(f"  exp. smoothing:     {results['exp_smoothing'][-1]:.2f}% MAPE")
        except Exception as e:
            print(f"  exp. smoothing FAILED: {e}")

        try:
            results["xgboost"].append(evaluate_xgboost(train, test))
            print(f"  xgboost:            {results['xgboost'][-1]:.2f}% MAPE")
        except Exception as e:
            print(f"  xgboost FAILED: {e}")

        try:
            results["prophet"].append(evaluate_prophet(train, test))
            print(f"  prophet:            {results['prophet'][-1]:.2f}% MAPE")
        except ImportError:
            print("  prophet not installed — `pip install prophet` to include it")
        except Exception as e:
            print(f"  prophet FAILED: {e}")

    print("\n=== Summary: mean MAPE across folds (lower is better) ===")
    for name, scores in results.items():
        if scores:
            print(f"{name:15s}: {np.mean(scores):6.2f}%  (n={len(scores)} folds)")
        else:
            print(f"{name:15s}: no successful folds")


if __name__ == "__main__":
    main()