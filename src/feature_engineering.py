"""Reusable feature engineering for CMAPSS turbofan RUL prediction.

The tree/linear models (Gradient Boosting, Random Forest, Ridge) were trained on
153 Min-Max-normalized engineered features derived from raw sensor readings.
The LSTM was trained on a 20-feature subset (the "top features"), also normalized.

This module rebuilds those exact feature spaces from raw sensor input so the
dashboard can run real predictions on new data instead of crashing on a
feature-count mismatch. The engineering formulas mirror
``CMAPSSPreprocessor`` (rolling windows, lags, trend/slope) and the Min-Max
parameters are the per-column min/max of the full FD001 training set, persisted
to ``feature_norm_params.json`` so inference normalizes identically to training.

Run as a script to (re)generate the normalization parameters:

    python src/feature_engineering.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Raw input columns (FD001 keeps these after cleaning; sensors 1,5,6,10,14,16,18,19
# were dropped earlier as constant/low-variance).
RAW_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21',
]
OPERATIONAL_SETTINGS = [
    'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
]
ID_COLS = ['engine_id', 'time_cycles']
TARGET_COLS = ['RUL', 'RUL_clipped']

WINDOWS = [5, 10, 20]
LAGS = [1, 3, 5]
TREND_WINDOW = 10

_BASE_DIR = Path(__file__).resolve().parent.parent
SILVER_FILE = _BASE_DIR / 'data' / 'silver' / 'cmapss' / 'FD001_cleaned.csv'
FEATURED_FILE = _BASE_DIR / 'data' / 'gold' / 'cmapss' / 'FD001_featured.csv'
TOP_FEATURES_FILE = _BASE_DIR / 'data' / 'gold' / 'cmapss' / 'FD001_top_features.csv'
NORM_PARAMS_FILE = _BASE_DIR / 'data' / 'models' / 'cmapss' / 'feature_norm_params.json'


def engineer_features(df: pd.DataFrame, unit_col: str = 'engine_id') -> pd.DataFrame:
    """Add rolling/lag/trend features to raw sensor data.

    Replicates ``CMAPSSPreprocessor.add_rolling_features`` /
    ``add_lag_features`` / ``add_trend_features`` exactly: rolling stats use
    ``min_periods=1``, lag gaps are forward/back-filled within each engine, and
    trend features (diff + rolling slope) fill NaN with 0. No EWMA, no
    normalization - those were not part of the trained feature set.
    """
    df = df.copy()
    sensors = [c for c in RAW_SENSORS if c in df.columns]

    for window in WINDOWS:
        for sensor in sensors:
            grouped = df.groupby(unit_col)[sensor]
            df[f'{sensor}_rolling_mean_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{sensor}_rolling_std_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)
            df[f'{sensor}_rolling_min_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'{sensor}_rolling_max_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).max())

    for lag in LAGS:
        for sensor in sensors:
            df[f'{sensor}_lag_{lag}'] = df.groupby(unit_col)[sensor].shift(lag)
    for col in [c for c in df.columns if '_lag_' in c]:
        df[col] = df.groupby(unit_col)[col].ffill().bfill()

    def _rolling_slope(series: pd.Series) -> float:
        if len(series) < 2:
            return 0.0
        y = series.values
        if np.std(y) == 0:
            return 0.0
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])

    for sensor in sensors:
        df[f'{sensor}_diff'] = df.groupby(unit_col)[sensor].diff()
        df[f'{sensor}_slope_{TREND_WINDOW}'] = df.groupby(unit_col)[sensor].transform(
            lambda x: x.rolling(window=TREND_WINDOW, min_periods=2).apply(
                _rolling_slope, raw=False))
    trend_cols = [c for c in df.columns if '_diff' in c or '_slope_' in c]
    df[trend_cols] = df[trend_cols].fillna(0)
    return df


def get_model_feature_columns() -> list[str]:
    """The 153 column names the tree/linear models were trained on, in order."""
    featured = pd.read_csv(FEATURED_FILE, nrows=0)
    return [c for c in featured.columns
            if c not in ID_COLS + TARGET_COLS]


def get_lstm_feature_columns() -> list[str]:
    """The 20 column names the LSTM was trained on, in order."""
    top = pd.read_csv(TOP_FEATURES_FILE, nrows=0)
    return [c for c in top.columns
            if c not in ID_COLS + TARGET_COLS]


def build_norm_params() -> dict:
    """Compute per-column min/max from the full FD001 training set.

    MinMaxScaler stores ``min_`` and ``scale_`` per column; for a [0, 1] range
    those are simply the column's min and (max - min). Fitting on the full
    training set reproduces the normalization the models were trained with.
    """
    cleaned = pd.read_csv(SILVER_FILE)
    engineered = engineer_features(cleaned)
    model_cols = get_model_feature_columns()
    params = {}
    for col in model_cols:
        if col in OPERATIONAL_SETTINGS:
            # Operational settings are not normalized by the training pipeline.
            continue
        values = engineered[col]
        col_min = float(values.min())
        col_max = float(values.max())
        params[col] = {'min': col_min, 'max': col_max}
    return params


def save_norm_params() -> Path:
    """Persist normalization parameters to disk (run once, e.g. after retraining)."""
    params = build_norm_params()
    NORM_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_PARAMS_FILE, 'w') as f:
        json.dump(params, f, indent=2)
    return NORM_PARAMS_FILE


def load_norm_params() -> dict:
    """Load persisted min/max parameters, building them on first use."""
    if not NORM_PARAMS_FILE.exists():
        save_norm_params()
    with open(NORM_PARAMS_FILE) as f:
        return json.load(f)


def _normalize(df: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    """Apply training min/max normalization in place to the given columns."""
    df = df.copy()
    for col in columns:
        if col in OPERATIONAL_SETTINGS or col not in params:
            continue
        mn = params[col]['min']
        mx = params[col]['max']
        if mx > mn:
            df[col] = (df[col] - mn) / (mx - mn)
        else:
            df[col] = 0.0
    return df


def prepare_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer + normalize raw sensor data into the 153 model feature columns.

    Returns a DataFrame with exactly the model's training columns, in order,
    suitable for ``GradientBoostingRegressor`` / ``RandomForestRegressor`` /
    ``Ridge`` ``.predict()``.
    """
    params = load_norm_params()
    engineered = engineer_features(raw_df)
    model_cols = get_model_feature_columns()
    normalized = _normalize(engineered, model_cols, params)
    # Select the exact training columns; fill any unforeseen gap with 0.
    for col in model_cols:
        if col not in normalized.columns:
            normalized[col] = 0.0
    return normalized[model_cols]


def prepare_lstm_sequence(raw_df: pd.DataFrame, sequence_length: int = 30) -> np.ndarray:
    """Engineer + normalize raw sensor data into an LSTM input sequence.

    Returns an array of shape ``(1, sequence_length, n_lstm_features)`` using
    the last ``sequence_length`` cycles of each engine. If an engine has fewer
    cycles, the start is padded by repeating its earliest cycle so the model
    always sees a full-length window (matching how it was trained).
    """
    params = load_norm_params()
    engineered = engineer_features(raw_df)
    lstm_cols = get_lstm_feature_columns()
    normalized = _normalize(engineered, lstm_cols, params)
    for col in lstm_cols:
        if col not in normalized.columns:
            normalized[col] = 0.0
    seq = normalized[lstm_cols].values

    if len(seq) >= sequence_length:
        seq = seq[-sequence_length:]
    elif len(seq) > 0:
        pad = np.repeat(seq[:1], sequence_length - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    else:
        seq = np.zeros((sequence_length, len(lstm_cols)))
    return seq.reshape(1, sequence_length, len(lstm_cols))


def make_steady_state_history(sensors: dict, n_cycles: int = 30) -> pd.DataFrame:
    """Build a steady-state engine history from manually entered sensor values.

    Manual entry provides a single cycle of readings. To compute rolling/lag/
    slope features we need history, so we synthesize ``n_cycles`` identical
    cycles at the entered values. This represents an engine that has been
    reading those stable values - a well-defined, interpretable input.
    """
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {'engine_id': 1, 'time_cycles': cycle}
        row.update(sensors)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    path = save_norm_params()
    print(f"Normalization parameters saved to {path}")
    print(f"Columns: {len(load_norm_params())}")
