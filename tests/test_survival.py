"""Survival tests.

Centred on the two things that were wrong: covariates that could see past the
prediction point, and scores computed on the data the model was fitted to.
"""

import numpy as np
import pandas as pd
import pytest

from survival import (HORIZON, LANDMARK, WINDOW, build_landmark_frame,
                      select_covariates)


@pytest.fixture(scope='module')
def landmark(train_fd001):
    return build_landmark_frame(train_fd001)


def test_one_row_per_eligible_engine(landmark, train_fd001):
    lifetimes = train_fd001.groupby('engine_id')['time_cycles'].max()
    expected = int((lifetimes > LANDMARK).sum())
    assert len(landmark) == expected
    assert landmark['engine_id'].is_unique


def test_covariates_cannot_see_past_the_landmark(train_fd001):
    """The core leakage test.

    Changing anything after the landmark must leave every covariate identical.
    The old design summarised the last 30 cycles before failure, so the
    covariates encoded exactly the thing being predicted.
    """
    baseline = build_landmark_frame(train_fd001)

    tampered = train_fd001.copy()
    after = tampered['time_cycles'] > LANDMARK
    sensor_cols = [c for c in tampered.columns if c.startswith('sensor_')]
    tampered.loc[after, sensor_cols] *= 1000

    result = build_landmark_frame(tampered)

    covariate_cols = [c for c in baseline.columns
                      if c not in ('engine_id', 'duration', 'event')]
    pd.testing.assert_frame_equal(result[covariate_cols], baseline[covariate_cols])


def test_duration_is_time_remaining_after_the_landmark(landmark, train_fd001):
    lifetimes = train_fd001.groupby('engine_id')['time_cycles'].max()
    for _, row in landmark.head(20).iterrows():
        remaining = lifetimes[row['engine_id']] - LANDMARK
        assert row['duration'] == min(remaining, HORIZON)
        assert row['event'] == int(remaining < HORIZON)


def test_events_are_censored_at_the_horizon(landmark):
    """Event must not be a constant.

    It was 1 for every engine before, which makes the survival machinery a very
    elaborate way of doing regression.
    """
    assert landmark['event'].nunique() == 2
    assert (landmark['duration'] <= HORIZON).all()

    # Censored observations all sit exactly at the horizon: that is where we
    # stopped watching, not where anything happened.
    censored = landmark[landmark['event'] == 0]
    assert len(censored) > 0
    assert (censored['duration'] == HORIZON).all()

    # Observed failures happened strictly before we stopped watching.
    failed = landmark[landmark['event'] == 1]
    assert len(failed) > 0
    assert (failed['duration'] < HORIZON).all()


def test_covariate_window_respects_its_width(train_fd001):
    """A narrower window must give different covariates than a wider one."""
    narrow = build_landmark_frame(train_fd001, window=5)
    wide = build_landmark_frame(train_fd001, window=WINDOW)
    col = 'sensor_2_mean'
    assert not np.allclose(narrow[col], wide[col])


def test_last_value_is_the_landmark_cycle(train_fd001, landmark):
    """The `_last` covariate should be the reading at the landmark itself."""
    engine = train_fd001[(train_fd001['engine_id'] == 1)
                         & (train_fd001['time_cycles'] == LANDMARK)]
    expected = float(engine['sensor_2'].iloc[0])
    actual = float(landmark.loc[landmark['engine_id'] == 1, 'sensor_2_last'].iloc[0])
    assert actual == pytest.approx(expected)


def test_covariate_selection_uses_only_the_frame_given(landmark):
    chosen = select_covariates(landmark, max_features=6)
    assert len(chosen) == 6
    assert 'duration' not in chosen and 'event' not in chosen
    assert 'engine_id' not in chosen


def test_models_are_scored_out_of_sample():
    """concordance() must not just return the fitted attribute.

    Reporting `concordance_index_` as model performance was the original bug,
    and it is easy to reintroduce because it is the convenient thing to reach for.
    """
    pytest.importorskip('lifelines')
    from splits import make_splits
    from survival import concordance, fit_survival_models

    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    frame = pd.DataFrame({
        'engine_id': np.arange(1, n + 1),
        'duration': np.clip(60 + 20 * x + rng.normal(scale=5, size=n), 5, 125),
        'event': 1,
        'feature_a': x,
        'feature_b': rng.normal(size=n),
    })
    frame.loc[frame['duration'] >= 125, 'event'] = 0

    splits = make_splits(frame['engine_id'].to_numpy())
    train = frame[frame['engine_id'].isin(splits['train'])]
    test = frame[frame['engine_id'].isin(splits['test'])]

    models, _, design = fit_survival_models(train, ['feature_a', 'feature_b'])
    weibull = models['weibull']

    held_out = concordance(weibull, design(test))
    assert 0.0 <= held_out <= 1.0
    # The signal is strong here, so held out concordance should be well above
    # chance; the assertion is that it is computed at all, on the test frame.
    assert held_out > 0.6
    assert held_out != pytest.approx(weibull.concordance_index_, abs=1e-9)
