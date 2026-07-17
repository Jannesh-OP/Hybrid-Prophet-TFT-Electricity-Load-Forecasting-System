# Hybrid Prophet + Temporal Fusion Transformer for Electricity Load Forecasting

A hybrid deep learning pipeline for short-term (24-hour ahead) electricity demand forecasting, combining **Prophet** as a rolling feature extractor with a **Temporal Fusion Transformer (TFT)** as the primary forecasting model. Built and evaluated on real-world hourly electricity demand and weather data from Panama.

## Overview

Pure deep learning models can struggle to capture long-term trend and calendar seasonality cleanly, while classical time-series models like Prophet don't model complex non-linear interactions between exogenous variables well. This project addresses that gap with a two-stage hybrid approach:

1. **Prophet** is fit on a rolling/walk-forward basis (refit periodically using only past data) to decompose the load series into trend and seasonality components — these are extracted as engineered features rather than used as final forecasts.
2. A **Temporal Fusion Transformer** (via `pytorch-forecasting`) then learns from the raw load, lag features, calendar features, weather covariates, and the Prophet-derived trend/seasonality signals to produce probabilistic 24-hour-ahead forecasts (P10/P50/P90 quantiles).

This mirrors the methodology of published hybrid statistical + deep learning forecasting research.

## Dataset

- **Source:** [Hourly Energy Demand, Generation & Weather — Panama](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting) (Kaggle)
- **Granularity:** Hourly, national electricity demand (`nat_demand`) for Panama, alongside weather variables (temperature, humidity, cloud cover, wind speed) from three cities/stations: Tocumen (`toc`), Santiago (`san`), and David (`dav`).
- **Span:** ~2015–2020 (2020 excluded from modeling to avoid pandemic-related demand anomalies).
- The dataset is **not included in this repo** due to size/licensing — see [Setup](#setup) for how to get it.

## Methodology

### 1. Preprocessing & Feature Engineering
- Renamed and cleaned raw columns; parsed and sorted timestamps.
- Calendar features: month, weekend flag, season.
- Cyclical encodings for hour and month (`sin`/`cos`) so e.g. hour 23 and hour 0 are treated as adjacent.
- Lag features: `t-1`, `t-24` (previous day), `t-168` (previous week, same hour).
- IQR-based outlier removal on the load series.
- 2020 data dropped to avoid pandemic-driven demand shocks distorting the learned patterns.

### 2. Prophet as a Rolling Feature Extractor
- Prophet is refit at multiple rolling origins (walk-forward), generating trend/seasonality only for the *next* window of data each time — this strictly prevents future information from leaking into past feature values.
- Includes Panama public holidays as a Prophet regressor.
- Output: `prophet_trend` and `prophet_seasonality` columns, merged back onto the main dataframe as model features.

### 3. Temporal Fusion Transformer
- Implemented with `pytorch-forecasting` on top of PyTorch Lightning.
- Encoder length: 168 hours (1 week lookback) → Decoder/prediction length: 24 hours.
- Chronological 70% / 15% / 15% train / validation / test split.
- Quantile loss (`QuantileLoss`, quantiles 0.1 / 0.5 / 0.9) for probabilistic forecasting.
- Trained across multiple random seeds (42, 123, 7) for robustness; best checkpoint selected on validation loss.

### 4. Evaluation & Interpretation
- Point-forecast metrics: MAE, MSE, RMSE, MAPE (computed on the P50 median forecast).
- Probabilistic evaluation via the P10/P90 quantile band.
- Model interpretation via TFT's built-in attention/variable-importance interpretation.
- Visual diagnostics: actual vs. predicted overlays, weekly demand profiles, seasonal/day-of-week load patterns.

## Results

| Metric | Value |
|---|---|
| MAE | 48.12 MW |
| RMSE | 62.13 MW |
| MAPE | 3.80% |
| Approx. Accuracy (100 − MAPE) | 96.20% |

*(Evaluated on the held-out chronological test set, April–December 2019.)*

## Tech Stack

- **Modeling:** `prophet`, `pytorch-forecasting`, `pytorch-lightning` (`lightning.pytorch`), PyTorch
- **Interpretability:** TFT native attention interpretation
- **Data handling & viz:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (metrics)
- **Environment:** Kaggle Notebook (GPU runtime, T4)



## Key Design Choices / Notes

- **No data leakage in Prophet features:** because Prophet is refit on a rolling/walk-forward basis and only generates features for future windows relative to its fit point, the trend/seasonality features are safe to use as model inputs without leaking test-period information into training.
- **Pandemic period excluded:** 2020 was removed since COVID-era demand shocks are a distribution shift that isn't representative of normal seasonal/weekly patterns, and including it would hurt generalization to typical years.
- **Probabilistic, not just point forecasts:** quantile loss lets the model express forecast uncertainty (P10–P90 band), which is more realistic for operational load forecasting than a single point estimate.

## Future Improvements

- Extend to multi-region/multi-series forecasting using the 3 city-level weather series directly.
- Hyperparameter tuning via Optuna (learning rate, hidden size, attention heads).
- Compare against baselines (ARIMA/SARIMA, LightGBM, vanilla LSTM) to better contextualize the hybrid model's gains.
- Deploy as a simple API/dashboard for interactive forecasting.

## Author

Evans — M.Tech Data Science, Delhi Technological University (DTU)
