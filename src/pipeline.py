"""Fitted feature pipeline: one object that owns the whole raw-to-model-matrix path.

Why this exists
---------------
Feature engineering used to be split across two places that had to agree by
hand: ``CMAPSSPreprocessor`` built the training matrix, and a second module
rebuilt what it guessed were the same columns at inference time. They drifted.

It also fit its transforms at the wrong moment. ``normalize_features`` and
``remove_correlated_features`` ran over all 100 engines before any split
existed, so the scaler's min/max and the surviving feature set both carried
information from the validation and test engines.

This class fixes both. The stateless part (rolling windows, lags, trend, EWMA)
is applied to any frame. The stateful part (which features survive the
correlation filter, and the min/max used to normalize them) is fit on training
engines only and then persisted. Inference loads the same fitted object, so the
serving path cannot drift from the training path by construction.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocessor import CMAPSSPreprocessor

_BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_FILE = _BASE_DIR / 'data' / 'models' / 'cmapss' / 'feature_pipeline.joblib'

ID_COLS = ['engine_id', 'time_cycles']
TARGET_COLS = ['RUL', 'RUL_clipped']


@dataclass
class FeaturePipeline:
    """Turns raw cleaned sensor cycles into the model matrix.

    Attributes set by ``fit`` carry a trailing underscore, following the
    scikit-learn convention for learned state.
    """

    windows: list = field(default_factory=lambda: [5, 10, 20])
    lags: list = field(default_factory=lambda: [1, 3, 5])
    trend_window: int = 10
    ewma_spans: list = field(default_factory=lambda: [5, 10, 20])
    corr_threshold: float = 0.95
    rul_clip: int = 125

    sensor_cols_: list | None = None
    feature_names_: list | None = None
    dropped_correlated_: list | None = None
    feature_min_: pd.Series | None = None
    feature_range_: pd.Series | None = None

    # -- stateless part ----------------------------------------------------

    def build_features(self, df: pd.DataFrame, unit_col: str = 'engine_id',
                       verbose: bool = False) -> pd.DataFrame:
        """Add rolling, lag, trend and EWMA features. No fitting, no leakage.

        Every feature here is computed within a single engine's own history, so
        applying this before the split is safe: no row can see another engine.
        """
        prep = CMAPSSPreprocessor(df)
        if self.sensor_cols_ is None:
            self.sensor_cols_ = list(prep.sensor_cols)
        else:
            missing = [c for c in self.sensor_cols_ if c not in df.columns]
            if missing:
                raise ValueError(f'Input is missing sensor columns: {missing}')
            prep.sensor_cols = list(self.sensor_cols_)

        with _quiet(verbose):
            out = prep.add_rolling_features(df, self.windows, unit_col)
            out = prep.add_lag_features(out, self.lags, unit_col)
            out = prep.add_trend_features(out, self.trend_window, unit_col)
            out = prep.add_ewma_features(out, self.ewma_spans, unit_col)
        return out

    def add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the clipped RUL target.

        Clipping is a modelling choice about the label, not something learned
        from the data, so it does not need to be fit on train only.
        """
        out = df.copy()
        out['RUL_clipped'] = out['RUL'].clip(upper=self.rul_clip)
        return out

    # -- stateful part -----------------------------------------------------

    def fit(self, train_df: pd.DataFrame,
            already_featured: bool = False) -> 'FeaturePipeline':
        """Learn the feature set and the normalization statistics.

        ``train_df`` must contain training engines only. Everything learned
        here is derived from those rows and nothing else.
        """
        featured = train_df if already_featured else self.build_features(train_df)
        candidates = [c for c in featured.columns
                      if c not in ID_COLS + TARGET_COLS]

        self.dropped_correlated_ = self._find_correlated(featured[candidates])
        self.feature_names_ = [c for c in candidates
                               if c not in self.dropped_correlated_]

        kept = featured[self.feature_names_]
        self.feature_min_ = kept.min()
        self.feature_range_ = (kept.max() - kept.min()).replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame, already_featured: bool = False) -> pd.DataFrame:
        """Apply the fitted feature set and normalization.

        Returns exactly ``feature_names_``, in order, so a model trained on the
        output of this method can always consume it.
        """
        self._check_fitted()
        featured = df if already_featured else self.build_features(df)

        missing = [c for c in self.feature_names_ if c not in featured.columns]
        if missing:
            raise ValueError(
                f'{len(missing)} feature(s) missing after engineering, '
                f'first few: {missing[:5]}')

        kept = featured[self.feature_names_]
        normalized = (kept - self.feature_min_) / self.feature_range_
        return normalized

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    # -- persistence -------------------------------------------------------

    def save(self, path: Path = PIPELINE_FILE) -> Path:
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = PIPELINE_FILE) -> 'FeaturePipeline':
        if not path.exists():
            raise FileNotFoundError(
                f'No fitted pipeline at {path}. Run `python -m src.train` first.')
        return joblib.load(path)

    # -- internals ---------------------------------------------------------

    def _find_correlated(self, features: pd.DataFrame) -> list:
        """Drop one of each pair correlated above the threshold.

        Keeps the first column of each pair, matching the original behaviour.
        The point of doing this on training rows only is that "which features
        are redundant" is itself a decision learned from data.
        """
        if features.columns.duplicated().any():
            dupes = features.columns[features.columns.duplicated()].unique().tolist()
            raise ValueError(f'Duplicate feature columns: {dupes[:5]}')
        corr = features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return [c for c in upper.columns if (upper[c] > self.corr_threshold).any()]

    def _check_fitted(self):
        if self.feature_names_ is None:
            raise RuntimeError('Pipeline is not fitted. Call fit() or load().')


def _quiet(verbose: bool):
    """Silence the preprocessor's progress prints unless asked for."""
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())
