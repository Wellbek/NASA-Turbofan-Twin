"""Inference side feature preparation.

This module used to contain a second, hand maintained copy of the feature
formulas plus its own min/max statistics recomputed from the full dataset. That
is the train/inference mismatch in #10: two implementations that had to be kept
in step by hand, and normalization statistics that did not match the ones the
models were actually fitted with.

Now there is one implementation. ``FeaturePipeline`` is fitted during training
and saved; everything here loads that object and applies it. If the training
feature set changes, this path changes with it, because it is the same object.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import FeaturePipeline

# Raw input columns. FD001 keeps these after cleaning; sensors 1, 5, 6, 10, 14,
# 16, 18 and 19 are dropped earlier as constant or low variance.
RAW_SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21',
]
OPERATIONAL_SETTINGS = [
    'operational_setting_1', 'operational_setting_2', 'operational_setting_3',
]

_BASE_DIR = Path(__file__).resolve().parent.parent
LSTM_SCALER_FILE = _BASE_DIR / 'data' / 'models' / 'cmapss' / 'lstm_scaler.joblib'
LSTM_FEATURES_FILE = _BASE_DIR / 'data' / 'models' / 'cmapss' / 'lstm_features.json'

_pipeline: FeaturePipeline | None = None


def get_pipeline() -> FeaturePipeline:
    """The fitted pipeline, loaded once per process."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline.load()
    return _pipeline


def get_model_feature_columns() -> list[str]:
    """Column names the tree and linear models were trained on, in order."""
    return list(get_pipeline().feature_names_)


def get_lstm_feature_columns() -> list[str]:
    """Column names the LSTM was trained on, in order."""
    with open(LSTM_FEATURES_FILE) as f:
        return json.load(f)['features']


def get_lstm_sequence_length() -> int:
    with open(LSTM_FEATURES_FILE) as f:
        return json.load(f)['sequence_length']


def prepare_model_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer and normalize raw sensor cycles into the model matrix.

    Returns exactly the training columns, in training order, ready for
    ``.predict()`` on any of the scikit-learn models.
    """
    return get_pipeline().transform(raw_df)


def prepare_lstm_sequence(raw_df: pd.DataFrame,
                          sequence_length: int | None = None) -> np.ndarray:
    """Build one LSTM input window of shape ``(1, sequence_length, n_features)``.

    Applies the same two step scaling the model was trained with: the pipeline's
    min/max normalization, then the ``StandardScaler`` fitted on the training
    sequences. That second step is what used to be missing, because the scaler
    was never saved and inference silently skipped it.
    """
    import joblib

    if sequence_length is None:
        sequence_length = get_lstm_sequence_length()

    features = prepare_model_features(raw_df)
    lstm_cols = get_lstm_feature_columns()
    seq = features[lstm_cols].to_numpy()

    if len(seq) >= sequence_length:
        seq = seq[-sequence_length:]
    elif len(seq) > 0:
        # Pad the start by repeating the earliest cycle so the model always
        # sees a full window, matching how it was trained.
        pad = np.repeat(seq[:1], sequence_length - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    else:
        seq = np.zeros((sequence_length, len(lstm_cols)))

    scaler = joblib.load(LSTM_SCALER_FILE)
    seq = scaler.transform(seq)
    return seq.reshape(1, sequence_length, len(lstm_cols))


def make_steady_state_history(sensors: dict, n_cycles: int = 30) -> pd.DataFrame:
    """Build an engine history from a single set of manually entered readings.

    Manual entry gives one cycle, but rolling, lag and slope features need
    history. Repeating the entered values for ``n_cycles`` represents an engine
    that has been holding those readings steady, which is well defined and
    interpretable, though it does mean every trend feature comes out flat.
    """
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {'engine_id': 1, 'time_cycles': cycle}
        row.update(sensors)
        rows.append(row)
    return pd.DataFrame(rows)
