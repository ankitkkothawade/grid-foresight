

from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import xgboost as xgb
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR.parent / "data" / "raw" / "demand_combined.csv"

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7
LATITUDE, LONGITUDE = 51.5072, -0.1276  # London, same proxy as step 2

# None = auto (day after the last date in your data). Or set e.g.
# pd.Timestamp("2026-08-10") to target a specific day.
TARGET_DATE = None


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DEMAND_FILE, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["lag_1day"] = df["ND"].shift(PERIODS_PER_DAY)
    df["lag_1week"] = df["ND"].shift(PERIODS_PER_WEEK)
    return df


def train_model(df: pd.DataFrame):
    features = ["hour", "day_of_week", "month", "is_weekend", "lag_1day", "lag_1week"]
    df_fe = make_features(df).dropna(subset=features)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(df_fe[features], df_fe["ND"])
    return model, features


def forecast_target_day(df: pd.DataFrame, model, features, target_date: pd.Timestamp) -> pd.DataFrame:
    periods = pd.date_range(target_date, periods=PERIODS_PER_DAY, freq="30min")
    future = pd.DataFrame({"datetime": periods})

    combined = pd.concat([df[["datetime", "ND"]], future], ignore_index=True)
    combined_fe = make_features(combined)
    future_fe = combined_fe.tail(PERIODS_PER_DAY)

    future["ND_forecast"] = model.predict(future_fe[features])
    return future


def fetch_archive_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical weather -- same endpoint verified in step 2."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m", "timezone": "Europe/London",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature_c": data["hourly"]["temperature_2m"],
    })


def retrieve_context(df: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    """The 'retrieval' half of RAG: pull real numbers to ground the LLM's
    explanation instead of letting it invent context."""
    context = {}

    last_week = target_date - timedelta(days=7)
    last_week_data = df[df["datetime"].dt.date == last_week.date()]
    context["last_week_avg_demand"] = last_week_data["ND"].mean() if len(last_week_data) else None

    # Forecast temperature for the target day itself. This uses Open-Meteo's
    # *forecast* endpoint (different from step 2's archive one) since this
    # syntax is less independently verified by me than the archive call --
    # if it errors, check Open-Meteo's current docs for this endpoint.
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": LATITUDE, "longitude": LONGITUDE,
                "start_date": target_date.strftime("%Y-%m-%d"),
                "end_date": target_date.strftime("%Y-%m-%d"),
                "hourly": "temperature_2m", "timezone": "Europe/London",
            },
            timeout=30,
        )
        resp.raise_for_status()
        temps = resp.json()["hourly"]["temperature_2m"]
        context["target_day_avg_temp"] = sum(temps) / len(temps)
    except Exception as e:
        context["target_day_avg_temp"] = None
        print(f"(Couldn't fetch target-day temperature: {e})")

    # Seasonal average: same calendar month, one year earlier, from the
    # archive API (always historical, so always available).
    try:
        month_start = (target_date - pd.DateOffset(years=1)).replace(day=1)
        month_end = month_start + pd.DateOffset(months=1)
        hist = fetch_archive_weather(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        context["seasonal_avg_temp"] = hist["temperature_c"].mean()
    except Exception as e:
        context["seasonal_avg_temp"] = None
        print(f"(Couldn't fetch seasonal average temperature: {e})")

    context["is_weekend"] = target_date.dayofweek >= 5
    context["day_name"] = target_date.day_name()
    return context


def generate_briefing(target_date, forecast_df, context) -> str:
    llm = ChatOllama(model="llama3.2", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a grid demand analyst writing a short daily briefing. "
         "Use ONLY the numbers given below. NEVER do arithmetic yourself -- "
         "the only two comparisons you are allowed to make are the two "
         "explicitly given below (marked with ->). Do NOT compute or state "
         "any OTHER difference (e.g. peak vs. average) and do NOT make any "
         "predictive or causal claim about what one number 'would suggest' "
         "about another -- state only what is given. If a value says 'not "
         "available', don't mention it. One tight paragraph."),
        ("human",
         "Date: {day_name}, {date}\n"
         "Forecast peak demand: {peak_demand:.0f} MW at {peak_time}\n"
         "Forecast average demand: {avg_demand:.0f} MW\n"
         "Same day last week's average demand: {last_week_avg}\n"
         "-> Average demand vs. last week: {demand_vs_last_week}\n"
         "Forecast average temperature: {target_temp}\n"
         "Typical seasonal average temperature for this time of year: {seasonal_temp}\n"
         "-> Temperature vs. seasonal norm: {temp_vs_seasonal}\n"
         "Weekend: {is_weekend}\n\n"
         "Write a short briefing explaining the forecast, using the two "
         "pre-computed comparisons above rather than calculating your own."),
    ])

    peak_idx = forecast_df["ND_forecast"].idxmax()
    avg_demand = forecast_df["ND_forecast"].mean()

    # Pre-compute every comparison here, in code, rather than asking the
    # LLM to subtract -- it got this wrong (1,122 instead of the real 4,898
    # MW gap) even with the correct numbers right in front of it.
    if context["last_week_avg_demand"]:
        delta = avg_demand - context["last_week_avg_demand"]
        demand_vs_last_week = f"{abs(delta):.0f} MW {'higher' if delta >= 0 else 'lower'} than last week"
    else:
        demand_vs_last_week = "not available"

    if context["target_day_avg_temp"] is not None and context["seasonal_avg_temp"] is not None:
        temp_delta = context["target_day_avg_temp"] - context["seasonal_avg_temp"]
        temp_vs_seasonal = f"{abs(temp_delta):.1f}\u00b0C {'above' if temp_delta >= 0 else 'below'} the seasonal average"
    else:
        temp_vs_seasonal = "not available"

    chain = prompt | llm
    response = chain.invoke({
        "day_name": context["day_name"],
        "date": target_date.strftime("%Y-%m-%d"),
        "peak_demand": forecast_df["ND_forecast"].max(),
        "peak_time": forecast_df.loc[peak_idx, "datetime"].strftime("%H:%M"),
        "avg_demand": avg_demand,
        "last_week_avg": f"{context['last_week_avg_demand']:.0f} MW" if context["last_week_avg_demand"] else "not available",
        "demand_vs_last_week": demand_vs_last_week,
        "target_temp": f"{context['target_day_avg_temp']:.1f}\u00b0C" if context["target_day_avg_temp"] is not None else "not available",
        "seasonal_temp": f"{context['seasonal_avg_temp']:.1f}\u00b0C" if context["seasonal_avg_temp"] is not None else "not available",
        "temp_vs_seasonal": temp_vs_seasonal,
        "is_weekend": "yes" if context["is_weekend"] else "no",
    })
    return response.content


def main():
    df = load_data()
    target_date = TARGET_DATE or (df["datetime"].max().normalize() + timedelta(days=1))
    print(f"Generating briefing for {target_date.date()}...\n")

    model, features = train_model(df)
    forecast_df = forecast_target_day(df, model, features, target_date)

    print("=== Forecast ===")
    print(f"Peak: {forecast_df['ND_forecast'].max():.0f} MW")
    print(f"Average: {forecast_df['ND_forecast'].mean():.0f} MW\n")

    context = retrieve_context(df, target_date)
    print("=== Retrieved context ===")
    for k, v in context.items():
        print(f"{k}: {v}")

    print("\n=== LLM Briefing ===")
    print(generate_briefing(target_date, forecast_df, context))


if __name__ == "__main__":
    main()