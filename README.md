# NASA Turbofan Twin – Predictive Maintenance for Jet Engines

## Project Overview

This project focuses on **predictive maintenance of turbofan engines** using NASA’s **CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset**. The goal is to predict the **Remaining Useful Life (RUL)** of engines, identify early signs of failure, and provide actionable insights for maintenance scheduling.

By simulating real-world engine degradation through time-series sensor data, this project demonstrates how advanced **statistical models, machine learning, and deep learning** can transform raw operational data into **predictive insights**, helping reduce unplanned downtime, optimize maintenance costs, and improve safety.

The results are visualized via an **interactive Streamlit dashboard**, making it easier to interpret predictions, monitor engine health, and understand which sensor features drive model decisions.

**This project is for self-educational purposes only.**

---

## Dataset

**NASA CMAPSS (C-MAPSS1, C-MAPSS2, etc.)**

* Simulated turbofan engine degradation datasets.
* Includes **multivariate time-series sensor readings** for multiple engines until failure.
* Provides an excellent benchmark for **RUL prediction**, enabling both **supervised regression** and **survival analysis** experiments.

---

## Why These Insights Matter

1. **Operational Efficiency**: Accurate RUL prediction allows maintenance teams to **schedule repairs just in time**, avoiding both unnecessary inspections and catastrophic failures.
2. **Safety Assurance**: Early detection of engine degradation reduces the risk of **in-flight failures**.
3. **Data-Driven Insights**: Feature importance and survival curves could help engineers understand **which sensors are most critical** for monitoring engine health.

---

## Tools & Libraries

* **Data Handling & EDA**: `pandas`, `numpy`, `matplotlib`, `seaborn` – for cleaning, visualizing, and exploring sensor data.
* **Statistical Survival Models**: `lifelines` – fit Weibull and Cox proportional hazards models to estimate engine failure probability over time.
* **Machine Learning**: `scikit-learn` – Random Forests, Gradient Boosting, and Ridge Regression for RUL prediction.
* **Deep Learning**: `TensorFlow / Keras` – LSTM and sequence models to capture temporal patterns in sensor readings.
* **Visualization & Dashboard**: `Streamlit`, `Plotly` – create interactive dashboards to explore engine health, feature importance, and RUL forecasts.

---

## 8-Week Project Pipeline

### 1. Data Ingestion & Exploration

* Load CMAPSS datasets using a custom loader.
* Perform exploratory analysis to understand sensor ranges, distributions, and patterns.
* **Why:** Before modeling, it’s crucial to understand the dataset and identify trends, anomalies, and preprocessing needs.

### 2. Feature Engineering

* Generate rolling-window features (mean, std, min, max, trends) for each sensor.
* Normalize sensor readings to handle scale differences.
* **Why:** Time-series features capture degradation patterns that single-time-point values cannot.

### 3. Baseline Models

* Train **linear regression and simple ML models** for RUL prediction.
* Evaluate baseline performance using MAE, RMSE, and R².
* **Why:** Provides a reference point to compare more complex models.

### 4. Survival Analysis

* Fit **Weibull and Cox models** to estimate failure probabilities over time.
* Produce survival curves for engines at different operational stages.
* **Why:** Offers probabilistic insights into engine health and complements deterministic RUL predictions.

### 5. Machine Learning Models

* Train **Random Forest and Gradient Boosting models** for RUL regression.
* Analyze **feature importance** to identify which sensors most influence predictions.
* **Why:** ML models capture nonlinear relationships and improve predictive accuracy over linear baselines.

### 6. Deep Learning Models

* Build **LSTM-based sequence models** to capture temporal dependencies in sensor data.
* Optionally explore **Transformer-based architectures** for improved long-range sequence modeling.
* **Why:** Temporal deep learning models can exploit sequential patterns in degradation that simpler models may miss.

### 7. Dashboard & Visualization

* Deploy a **Streamlit dashboard** to:

  * Display predicted RUL for individual engines.
  * Plot survival curves and prediction intervals.
  * Visualize sensor importance and trends.
* **Why:** Makes predictive insights **accessible and actionable** for project stakeholders, even without engineering experience.

---

## Expected Outcomes

* **High-quality RUL predictions** with benchmarked performance across linear, ML, and deep learning models.
* **Interpretability insights** via feature importance and survival analysis.
* **Interactive dashboard** for exploring engine health and predictive maintenance schedules.
* Reproducible, modular pipeline that can be extended to other turbofan datasets or industrial equipment in real-life scenarios.

---

## Evaluation Metrics

* **RUL Regression**:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * R² Score
  * Stratified evaluation by short-, medium-, and long-term horizons.
* **Failure Probability (Survival Analysis)**:

  * Survival curves
  * Predicted probability of failure over time
* **Uncertainty Estimates**:

  * Prediction intervals from Monte Carlo dropout or ensemble methods

