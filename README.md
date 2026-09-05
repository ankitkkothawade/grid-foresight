# Grid Foresight

**UK electricity demand forecasting, benchmarked directly against the grid operator's own forecast, with an LLM briefing layer that explains the numbers instead of just reporting them.**

**[Live demo](https://grid-foresight-2hmlbndrwlmkeyz3kqxhwe.streamlit.app/)**

*[Add a screenshot or short GIF of the app here \u2014 the chart + briefing panel together is the strongest single image for this project]*

---

## The headline result

| Model | MAPE (walk-forward validated, 4 folds) |
|---|---|
| Naive (same half-hour, 7 days prior) | 8.20% |
| Exponential Smoothing (damped) | 26.11% |
| Prophet | 16.08% |
| **XGBoost (lag + calendar features)** | **5.87%** |
| **NESO's own day-ahead forecast** | **3.92%** |

XGBoost beats a naive 7-day-lag baseline by 28% and comfortably outperforms both Prophet and Exponential Smoothing \u2014 using only public, historical data. It doesn't beat NESO's own operational forecast, and it shouldn't be expected to: NESO's forecasting team has access to submitted generation schedules, live weather forecasts (not just historical temperature), and embedded solar/wind output that this project structurally can't see. Landing within ~2 percentage points of a national grid operator's forecast, using a small fraction of their data access, is the actual result \u2014 not a leaderboard score to inflate, but a specific, explainable gap.

## What this project does

1. Pulls 5+ years of half-hourly Great Britain electricity demand from the [NESO Data Portal](https://www.neso.energy/data-portal/historic-demand-data).
2. Builds and walk-forward validates three forecasting approaches (Exponential Smoothing, Prophet, XGBoost) against a seasonal-naive baseline.
3. Benchmarks the best model directly against NESO's own published [day-ahead forecast performance data](https://www.neso.energy/data-portal/day-ahead-half-hourly-demand-forecast-performance/day_ahead_half_hourly_demand_forecast_performance) \u2014 same dates, same methodology, real apples-to-apples comparison.
4. Adds a **retrieval-augmented LLM layer**: retrieves real context (forecast temperature vs. the seasonal average, the same day's demand one week prior, weekend status) and asks an LLM to write a plain-English daily briefing \u2014 using *only* the retrieved numbers, never its own guesses.
5. Wraps the whole thing in an interactive Streamlit app: pick a date, see the forecast chart and the generated briefing together.

## Why this combination

Time-series forecasting and LLM/RAG are usually shown as separate portfolio projects. This one pairs them deliberately: the forecasting core has to be honestly validated (walk-forward, not a lucky train/test split) before the LLM layer is allowed to touch it, and the LLM layer is explicitly designed to be *grounded* \u2014 it explains a real, verified forecast rather than generating plausible-sounding text about a topic it has no real information on.

## Things I found and fixed along the way

A few specific, real problems came up during validation \u2014 documenting them here because diagnosing *why* something failed is more informative than just reporting a final number:

- **Exponential Smoothing initially produced nonsensical results (up to 635% MAPE).** Root cause: an undamped additive trend, extrapolated linearly over a 14-day forecast horizon, compounds and diverges. Fixed with `damped_trend=True` \u2014 a well-documented Holt-Winters failure mode, not a data problem.
- **A separate, unrelated-looking set of RuntimeWarnings ("divide by zero"/"overflow" in `matmul`) appeared in both Exponential Smoothing and Prophet.** Traced this to a confirmed cosmetic bug in NumPy 2.0.2's Accelerate BLAS backend on Apple Silicon \u2014 verified by reproducing the same warnings on completely clean synthetic data with no NaN/Inf in the actual output. Not a real numerical problem; silenced deliberately once confirmed.
- **The LLM briefing initially fabricated numbers**, even when given the correct data: it did its own (wrong) arithmetic on provided figures, and in one case invented a specific demand value with no basis in any retrieved data. Fixed by moving *all* arithmetic into Python and only ever handing the LLM pre-computed, labeled comparisons \u2014 plus an explicit instruction not to introduce any comparison beyond what's given. Verified this held up across multiple dates and two different models (Llama 3.2 locally via Ollama, then GPT-OSS-20B via Groq for the deployed version) before trusting it.

## Architecture

```
NESO Data Portal (historic demand, half-hourly)
        \u2193
  Data pipeline: merge years, dedupe, DST-aware datetime construction
        \u2193
  Feature engineering: calendar features + lag features (1 day, 1 week)
        \u2193
  Walk-forward validated models: naive \u2192 Exp. Smoothing \u2192 Prophet \u2192 XGBoost
        \u2193                                                        \u2193
  Benchmark vs. NESO's own forecast              Retrieval: weather (Open-Meteo),
                                                  last week's demand, holiday status
                                                            \u2193
                                                  LLM (Groq / GPT-OSS-20B) generates
                                                  a grounded plain-English briefing
        \u2193                                                        \u2193
                    Streamlit app: chart + metrics + briefing
```

## Tech stack

Python \u00b7 pandas \u00b7 XGBoost \u00b7 Prophet \u00b7 statsmodels \u00b7 LangChain \u00b7 Groq (GPT-OSS-20B) \u00b7 Streamlit \u00b7 Plotly \u00b7 [NESO Data Portal](https://www.neso.energy/data-portal) \u00b7 [Open-Meteo](https://open-meteo.com/)

## Running it locally

```bash
git clone <your-repo-url>
cd grid-foresight
pip install -r requirements.txt

export GROQ_API_KEY="your-free-key-from-console.groq.com"

streamlit run Code/grid_foresight_app.py
```

Data setup: download the historic demand year files from the [NESO Data Portal](https://www.neso.energy/data-portal/historic-demand-data) into `data/raw/`, then run `Code/grid_foresight_step1_data_pull.py` to build the combined, cleaned dataset the app reads.

## Limitations & possible next steps

- Weather is pulled for a single London coordinate as a proxy for GB-wide conditions, not a proper population-weighted national figure.
- The forecasting models use only historical demand + calendar features \u2014 no live weather forecast, no generation-side data, which is the main source of the gap to NESO's own numbers.
- Natural extensions identified during this project but not yet built: benchmarking against a pretrained time-series foundation model (e.g. Chronos-2) for zero-shot comparison, adding conformal prediction for calibrated uncertainty intervals instead of point forecasts, and a lightweight automated check that flags any LLM-generated number not traceable back to the retrieved context (a smaller version of hallucination-detection methods like SelfCheckGPT, applied directly to this pipeline).

## Data sources

- [NESO Data Portal](https://www.neso.energy/data-portal) \u2014 historic GB electricity demand and day-ahead forecast performance data. Published by the UK's National Energy System Operator.
- [Open-Meteo](https://open-meteo.com/) \u2014 free historical and forecast weather data, no API key required.
