

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import xgboost as xgb
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Make the API key work whether it comes from a local `export` (picked up
# via os.environ) or from Streamlit Cloud's Secrets manager (st.secrets) --
# ChatGroq itself only knows to look at the environment variable, so we
# copy it across explicitly rather than relying on any automatic linkage.
# st.secrets raises an error if NO secrets.toml exists anywhere at all
# (not just "key missing") -- expected during local testing via `export`,
# so this just falls back to whatever's already in the environment.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
DEMAND_FILE = SCRIPT_DIR.parent / "data" / "raw" / "demand_combined.csv"

PERIODS_PER_DAY = 48
PERIODS_PER_WEEK = PERIODS_PER_DAY * 7
LATITUDE, LONGITUDE = 51.5072, -0.1276  # London, same proxy as steps 2 and 5


@st.cache_data
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


@st.cache_resource
def train_model(_df: pd.DataFrame):
    # Leading underscore on the param tells Streamlit not to try hashing a
    # DataFrame for the cache key -- it just trains once and reuses it.
    features = ["hour", "day_of_week", "month", "is_weekend", "lag_1day", "lag_1week"]
    df_fe = make_features(_df).dropna(subset=features)
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
    model.fit(df_fe[features], df_fe["ND"])
    return model, features


def forecast_target_day(df, model, features, target_date) -> pd.DataFrame:
    periods = pd.date_range(target_date, periods=PERIODS_PER_DAY, freq="30min")
    future = pd.DataFrame({"datetime": periods})
    combined = pd.concat([df[["datetime", "ND"]], future], ignore_index=True)
    combined_fe = make_features(combined)
    future_fe = combined_fe.tail(PERIODS_PER_DAY)
    future["ND_forecast"] = model.predict(future_fe[features])
    return future


def fetch_archive_weather(start_date: str, end_date: str) -> pd.DataFrame:
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
    context = {}
    last_week = target_date - timedelta(days=7)
    last_week_data = df[df["datetime"].dt.date == last_week.date()]
    context["last_week_avg_demand"] = last_week_data["ND"].mean() if len(last_week_data) else None

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
    except Exception:
        context["target_day_avg_temp"] = None

    try:
        month_start = (target_date - pd.DateOffset(years=1)).replace(day=1)
        month_end = month_start + pd.DateOffset(months=1)
        hist = fetch_archive_weather(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
        context["seasonal_avg_temp"] = hist["temperature_c"].mean()
    except Exception:
        context["seasonal_avg_temp"] = None

    context["is_weekend"] = target_date.dayofweek >= 5
    context["day_name"] = target_date.day_name()
    return context


def generate_briefing(target_date, forecast_df, context) -> str:
    # llama-3.1-8b-instant was deprecated by Groq (July 2026 catalogue
    # update) -- gpt-oss-20b is one of their current recommended
    # replacements: compact and fast, a good fit for a short briefing task.
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)

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


# ---------------------------------------------------------------- UI ----

st.set_page_config(page_title="Grid Foresight", layout="wide")
st.title("\u26a1 Grid Foresight")
st.caption("UK electricity demand forecasting with an LLM briefing layer, "
           "benchmarked against NESO's own day-ahead forecast (5.87% MAPE "
           "vs. NESO's 3.92% -- see README).")

df = load_data()
model, features = train_model(df)

min_date = (df["datetime"].max().normalize() + timedelta(days=1)).date()
selected_date = st.date_input("Forecast date", value=min_date, min_value=min_date)
target_date = pd.Timestamp(selected_date)

if st.button("Generate forecast", type="primary"):
    with st.spinner("Forecasting..."):
        forecast_df = forecast_target_day(df, model, features, target_date)

    recent_actual = df[df["datetime"] >= target_date - timedelta(days=7)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_actual["datetime"], y=recent_actual["ND"],
                              mode="lines", name="Actual (last 7 days)"))
    fig.add_trace(go.Scatter(x=forecast_df["datetime"], y=forecast_df["ND_forecast"],
                              mode="lines", name="Forecast", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Demand (MW)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Forecast peak", f"{forecast_df['ND_forecast'].max():,.0f} MW")
    col2.metric("Forecast average", f"{forecast_df['ND_forecast'].mean():,.0f} MW")

    with st.spinner("Generating briefing (running locally via Ollama)..."):
        context = retrieve_context(df, target_date)
        briefing = generate_briefing(target_date, forecast_df, context)

    st.subheader("Daily briefing")
    st.write(briefing)

    with st.expander("Retrieved context (what grounded this briefing)"):
        st.json({k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in context.items()})