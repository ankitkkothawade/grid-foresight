
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Anchored to this script's own folder, matching where step 1 saves its
# output (<script's folder>/data/demand_combined.csv) — works regardless
# of what working directory your IDE runs the script from.
SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR.parent / "data" / "raw" / "demand_combined.csv"

# Representative single coordinate for a GB-wide weather proxy — London.
# This is a simplification: real GB demand responds to a population-weighted
# average temperature across the whole country, not one city. Good enough
# for a first correlation check; worth improving later (e.g. average a few
# cities, or use a proper population-weighted degree-day series) if you want
# to use temperature as a real model feature rather than just an EDA check.
LATITUDE, LONGITUDE = 51.5072, -0.1276


def load_demand() -> pd.DataFrame:
    df = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["month"] = df["datetime"].dt.month
    return df


def plot_seasonality(df: pd.DataFrame):
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    df.groupby("hour")["ND"].mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("Average demand by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Demand (MW)")

    df.groupby("day_of_week")["ND"].mean().reindex(dow_order).plot(kind="bar", ax=axes[1])
    axes[1].set_title("Average demand by day of week")
    axes[1].set_ylabel("Demand (MW)")

    df.groupby("month")["ND"].mean().plot(kind="bar", ax=axes[2])
    axes[2].set_title("Average demand by month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Demand (MW)")

    plt.tight_layout()
    plot_path = SCRIPT_DIR / "seasonality.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Pull hourly temperature from Open-Meteo's free historical archive
    (no API key required). Endpoint verified against Open-Meteo's own docs."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "Europe/London",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    weather = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
    })
    return weather


def correlate_with_weather(df: pd.DataFrame):
    start = df["datetime"].min().strftime("%Y-%m-%d")
    end = df["datetime"].max().strftime("%Y-%m-%d")
    print(f"\nFetching weather for {start} to {end}...")
    weather = fetch_weather(start, end)

    # Demand is half-hourly, weather is hourly — floor demand timestamps to
    # the hour so each half-hour joins to its containing hour's temperature.
    df = df.copy()
    df["hour_floor"] = df["datetime"].dt.floor("h")
    merged = df.merge(weather, left_on="hour_floor", right_on="datetime", suffixes=("", "_weather"))

    corr = merged["ND"].corr(merged["temperature_c"])
    print(f"Correlation between demand and temperature: {corr:.3f}")
    print("(Expect a negative correlation in a heating-dominated GB winter/")
    print(" summer mix — colder temperatures usually mean higher demand.)")

    plt.figure(figsize=(8, 6))
    plt.scatter(merged["temperature_c"], merged["ND"], s=2, alpha=0.3)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Demand (MW)")
    plt.title(f"Demand vs. temperature (corr = {corr:.3f})")
    plt.tight_layout()
    plot_path = SCRIPT_DIR / "demand_vs_temperature.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")

    return merged


def flag_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Flag UK public holidays. NESO's national demand figure isn't split
    by home nation, so this uses England as an approximation — worth
    checking the current `holidays` package docs for the exact class name
    and available subdivisions (e.g. Scotland-specific dates), since this
    library's API has changed across versions and I haven't independently
    re-verified the current syntax."""
    try:
        import holidays as holidays_pkg

        years = range(df["datetime"].dt.year.min(), df["datetime"].dt.year.max() + 1)
        uk_holidays = holidays_pkg.country_holidays("UK", subdiv="England", years=years)
        df = df.copy()
        df["is_holiday"] = df["datetime"].dt.date.astype(str).map(
            lambda d: pd.Timestamp(d).date() in uk_holidays
        )
        print(f"\nFlagged {df['is_holiday'].sum()} half-hour periods as holidays.")
    except Exception as e:
        print(f"\nCouldn't flag holidays automatically ({e}). "
              "Check the `holidays` package's current API — this step is "
              "optional for now and can be added later.")
        df = df.copy()
        df["is_holiday"] = False
    return df


def seasonal_naive_baseline(df: pd.DataFrame, test_days: int = 90):
    """Predict each half-hour as the same half-hour exactly 7 days earlier.
    Evaluate on the last `test_days` days of the dataset (held out)."""
    df = df.sort_values("datetime").reset_index(drop=True)

    periods_per_week = 48 * 7  # half-hours in a week
    df["naive_forecast"] = df["ND"].shift(periods_per_week)

    cutoff = df["datetime"].max() - pd.Timedelta(days=test_days)
    test = df[df["datetime"] > cutoff].dropna(subset=["naive_forecast"])

    ape = (test["ND"] - test["naive_forecast"]).abs() / test["ND"]
    mape = ape.mean() * 100

    print(f"\nSeasonal-naive baseline MAPE on last {test_days} days: {mape:.2f}%")
    print("Every model you build from here needs to beat this number.")
    return mape


def main():
    df = load_demand()
    plot_seasonality(df)

    try:
        correlate_with_weather(df)
    except Exception as e:
        print(f"\nWeather fetch failed ({e}) — check your internet connection "
              "or the Open-Meteo endpoint status. Skipping this step for now.")

    df = flag_holidays(df)
    seasonal_naive_baseline(df)


if __name__ == "__main__":
    main()