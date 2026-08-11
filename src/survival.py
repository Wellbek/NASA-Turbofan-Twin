"""Survival analysis on a landmark design, fit on training engines only.

What was wrong before
---------------------
Covariates were built from the last 30 cycles *before failure* and the target
was total lifetime. The sensor state at the moment of failure is not something
you have when the prediction is needed, so the model was reading the end of the
story to predict how long the story was.

Both models were also fit and scored on the same 100 engines, and
``concordance_index_`` is the *training* concordance by definition, so the 0.85
that was reported was never a generalisation number.

The event column was a constant 1. With nothing censored, survival analysis
collapses into ordinary regression with extra machinery, and none of the parts
that make it worth using were doing anything.

The landmark design
-------------------
Pick a landmark cycle ``L``. Every engine still running at ``L`` enters the
study, described only by what happened up to ``L``.

- **covariates**: summarised over ``(L - w, L]``, so nothing after the landmark
  is visible
- **duration**: cycles remaining after ``L``
- **event**: 1 if the engine fails within horizon ``H``, otherwise the
  observation is right censored at ``H`` and the event is 0

That is a question you can ask in service: this engine has run ``L`` cycles and
its sensors look like this, how much longer has it got. Censoring at ``H`` also
gives the event column real content, which is the reason to use a survival
model rather than a regressor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LANDMARK = 100          # cycles observed before the prediction is made
WINDOW = 30             # covariate window ending at the landmark
HORIZON = 125           # follow-up horizon, matching the RUL clip used elsewhere

_BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = _BASE_DIR / 'data' / 'models' / 'cmapss'


def build_landmark_frame(df: pd.DataFrame, sensor_cols: list | None = None,
                         landmark: int = LANDMARK, window: int = WINDOW,
                         horizon: int = HORIZON) -> pd.DataFrame:
    """One row per engine that survived to ``landmark``.

    Only cycles up to ``landmark`` are read. That constraint is the whole
    design, so it is enforced once by slicing the frame up front rather than
    trusted to every feature expression below.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]

    lifetimes = df.groupby('engine_id')['time_cycles'].max()
    eligible = lifetimes[lifetimes > landmark].index

    observed = df[df['engine_id'].isin(eligible) & (df['time_cycles'] <= landmark)]
    in_window = observed[observed['time_cycles'] > landmark - window]
    grouped = in_window.groupby('engine_id', sort=True)

    engine_ids = np.array(sorted(grouped.groups))
    remaining = lifetimes.loc[engine_ids].to_numpy() - landmark

    out = pd.DataFrame({
        'engine_id': engine_ids,
        # Right censor at the horizon: past that we simply stop watching. An
        # engine still running when we stop is censored, so the comparison is
        # strict and a failure exactly at the horizon counts as unobserved.
        'duration': np.minimum(remaining, horizon),
        'event': (remaining < horizon).astype(int),
    })

    features = {}
    for sensor in sensor_cols:
        features[f'{sensor}_mean'] = grouped[sensor].mean().to_numpy()
        features[f'{sensor}_last'] = grouped[sensor].last().to_numpy()
        # Drift across the window: where it ended minus where it started.
        features[f'{sensor}_drift'] = (grouped[sensor].last()
                                       - grouped[sensor].first()).to_numpy()

    return pd.concat([out, pd.DataFrame(features)], axis=1)


def select_covariates(train_frame: pd.DataFrame, max_features: int = 10) -> list:
    """Pick covariates by rank correlation with duration, on training rows.

    Both lifelines models are penalised but still struggle with dozens of
    collinear sensor summaries over 70 rows, so the set is kept deliberately
    small. Selection reads the frame it is given, which must be train only.
    """
    candidates = [c for c in train_frame.columns
                  if c not in ('engine_id', 'duration', 'event')]
    usable = [c for c in candidates if train_frame[c].std() > 1e-8]
    strength = (train_frame[usable]
                .corrwith(train_frame['duration'], method='spearman')
                .abs().sort_values(ascending=False))
    return strength.head(max_features).index.tolist()


def fit_survival_models(train_frame: pd.DataFrame, covariates: list,
                        penalizer: float = 0.1):
    """Fit Weibull AFT and Cox PH on the training engines.

    Covariates are standardised with training statistics only. The fitted
    standardiser comes back with the models because it is part of the artifact,
    not a local variable.
    """
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_frame[covariates])

    def design(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(scaler.transform(frame[covariates]),
                           columns=covariates, index=frame.index)
        out['duration'] = frame['duration'].to_numpy()
        out['event'] = frame['event'].to_numpy()
        return out

    waft = WeibullAFTFitter(penalizer=penalizer)
    waft.fit(design(train_frame), duration_col='duration', event_col='event')

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(design(train_frame), duration_col='duration', event_col='event')

    return {'weibull': waft, 'cox': cph}, scaler, design


def concordance(model, design_frame: pd.DataFrame) -> float:
    """Concordance index on the frame passed in.

    Deliberately not ``model.concordance_index_``: that attribute is the
    training concordance, and quoting it as model performance was the original
    mistake.
    """
    from lifelines.utils import concordance_index

    predicted = model.predict_expectation(design_frame)
    return float(concordance_index(design_frame['duration'], predicted,
                                   design_frame['event']))
