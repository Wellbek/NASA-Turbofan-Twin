# NASA Turbofan Twin - Predictive Maintenance & Demand Forecasting of NASA Jet Engines

## Overview

This project combines **predictive maintenance** (using NASA’s CMAPSS turbofan simulator) with **demand forecasting** (retail/manufacturing time-series + external macro indicators). The aim is to build end-to-end pipelines, from raw data ingestion to survival analysis, machine learning, and deep learning models, and deploy an interactive **Streamlit dashboard** showcasing results.

---

## Datasets & APIs

1. **Predictive Maintenance**

   * *NASA CMAPSS (C-MAPSS1, C-MAPSS2, etc.)* - simulated turbofan degradation time-series.

2. **Demand / Production Forecasting**

   * *Option A*: Kaggle “Store Item Demand Forecasting” dataset.
   * *Option B*: Synthetic monthly production dataset with trend + seasonality.

3. **External Indicators (APIs)**

   * **FRED API**: Industrial Production Index (IPI), PMI.
   * **Quandl / OECD / World Bank**: downloadable CSVs of industrial/manufacturing indices.

---

## Tools & Libraries

* **Data Handling & EDA**: pandas, numpy, matplotlib, seaborn
* **Classical Forecasting**: statsmodels (ARIMA/SARIMAX)
* **Machine Learning**: scikit-learn (Ridge, Random Forest, Gradient Boosting)
* **Survival Analysis**: lifelines (Weibull, Cox)
* **Deep Learning**: TensorFlow / Keras (LSTM, sequence models)
* **APIs & Integration**: requests, fredapi
* **Visualization & App**: Streamlit, Plotly
* **Deployment**: Dockerfile, Streamlit sharing

---

## 8-Week Project Plan

* **Week 1 - Data Ingestion & EDA**
  Download CMAPSS and demand datasets, implement loaders, run exploratory analysis.

* **Week 2 - Feature Engineering & Baselines**
  Create rolling-window sensor features; build Ridge regression for RUL and ARIMA for demand.

* **Week 3 - Statistical Survival Models**
  Fit Weibull / Cox models on CMAPSS; simulate demand/failure scenarios.

* **Week 4 - Machine Learning Models**
  Train Random Forest / Gradient Boosting for RUL. Analyze feature importances.

* **Week 5 - Deep Learning Demand Forecasting**
  Build LSTM/Transformer models with exogenous indicators (IPI, PMI).

* **Week 6 - Deep Survival Models**
  Train LSTM-based RUL regression or DeepSurv survival networks.

* **Week 7 - API Integration & Pipelines**
  Fetch data from FRED/Quandl, merge with demand data; finalize training/evaluation pipelines.

* **Week 8 - Dashboard & Finalization**
  Build Streamlit app: visualize RUL predictions, demand forecasts, uncertainty bands. Write final README, evaluation results, and record demo.

---

## Expected Outcomes

* A **reproducible pipeline** for predictive maintenance and demand forecasting.
* Benchmarks across statistical, ML, and DL models (MAE, RMSE, MAPE).
* **Interactive dashboard** with forecasts, survival curves, and feature importances.

---

## Evaluation

* **RUL Regression**: MAE, RMSE, R² (also stratified by short/medium/long horizon).
* **Failure Classification (optional)**: Precision, Recall, F1, AUC.
* **Demand Forecasting**: MAPE (primary), MAE, RMSE, with rolling-origin backtesting.
* **Uncertainty**: Quantile forecasts, Monte Carlo dropout.
