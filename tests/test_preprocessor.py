"""Preprocessor tests.

Mostly a guard against silent breakage from pandas deprecations: the groupby
fillna calls these exercise were dead for two minor releases before anyone
noticed, because nothing imported them outside a notebook.
"""

import numpy as np
import pytest

from preprocessor import CMAPSSPreprocessor


@pytest.fixture
def prep(small_slice):
    return CMAPSSPreprocessor(small_slice)


def test_constant_sensors_are_dropped(prep, small_slice):
    cleaned, dropped = prep.remove_constant_sensors(small_slice)
    assert dropped, 'FD001 has several constant sensors'
    assert not any(c in cleaned.columns for c in dropped)


def test_clip_rul_caps_at_the_threshold(prep, small_slice):
    out = prep.clip_rul(small_slice, clip_value=125)
    assert out['RUL_clipped'].max() <= 125
    # Values below the cap are untouched.
    below = out['RUL'] < 125
    assert (out.loc[below, 'RUL_clipped'] == out.loc[below, 'RUL']).all()


def test_rolling_features_have_expected_shape_and_no_nan(prep, small_slice):
    windows = [5, 10]
    before = small_slice.shape[1]
    out = prep.add_rolling_features(small_slice, windows=windows)
    added = len(windows) * len(prep.sensor_cols) * 4
    assert out.shape[1] == before + added
    # The caller's frame is left alone.
    assert small_slice.shape[1] == before
    new_cols = [c for c in out.columns if '_rolling_' in c]
    assert len(new_cols) == added
    assert out[new_cols].notna().all().all()


def test_lag_features_run_and_fill_the_first_cycles(prep, small_slice):
    """Regression test: this raised AttributeError on pandas 2.2+."""
    out = prep.add_lag_features(small_slice, lags=[1, 3])
    lag_cols = [c for c in out.columns if '_lag_' in c]
    assert len(lag_cols) == 2 * len(prep.sensor_cols)
    assert out[lag_cols].notna().all().all()


def test_lag_features_do_not_bleed_across_engines(prep, small_slice):
    sensor = prep.sensor_cols[0]
    out = prep.add_lag_features(small_slice, lags=[1])
    second_engine = sorted(out['engine_id'].unique())[1]
    first_row = out[out['engine_id'] == second_engine].iloc[0]
    # With nothing to look back on, the backfill uses this engine's own value.
    assert first_row[f'{sensor}_lag_1'] == first_row[sensor]


def test_trend_features_run_and_fill_the_first_cycle(prep, small_slice):
    """Regression test: this raised AttributeError on pandas 2.2+."""
    out = prep.add_trend_features(small_slice, window=5)
    trend_cols = [c for c in out.columns if c.endswith('_diff') or '_slope_' in c]
    assert len(trend_cols) == 2 * len(prep.sensor_cols)
    assert out[trend_cols].notna().all().all()

    sensor = prep.sensor_cols[0]
    first_row = out[out['engine_id'] == out['engine_id'].iloc[0]].iloc[0]
    assert first_row[f'{sensor}_diff'] == 0


@pytest.mark.parametrize('window', [5, 10, 20])
def test_rolling_slope_matches_a_least_squares_fit(small_slice, window):
    """The closed form slope must agree with an explicit polyfit.

    The closed form replaced a per-window np.polyfit call for speed, so this
    pins it against the thing it replaced rather than against stored numbers.
    """
    from preprocessor import _rolling_slope

    series = small_slice[small_slice['engine_id'] == 1]['sensor_2'].reset_index(drop=True)

    def reference(s):
        if len(s) < 2:
            return 0.0
        y = s.to_numpy()
        if np.std(y) == 0:
            return 0.0
        return np.polyfit(np.arange(len(y)), y, 1)[0]

    expected = series.rolling(window=window, min_periods=2).apply(reference, raw=False)
    actual = _rolling_slope(series, window)

    assert expected.isna().equals(actual.isna())
    np.testing.assert_allclose(actual.dropna(), expected.dropna(), atol=1e-9)


def test_rolling_slope_is_zero_for_a_flat_signal():
    from preprocessor import _rolling_slope
    import pandas as pd

    flat = pd.Series([3.0] * 40)
    assert float(np.nanmax(np.abs(_rolling_slope(flat, 10)))) == 0.0


def test_rolling_slope_recovers_a_known_gradient():
    from preprocessor import _rolling_slope
    import pandas as pd

    ramp = pd.Series(np.arange(50, dtype=float) * 2.5)
    slopes = _rolling_slope(ramp, 10).dropna()
    np.testing.assert_allclose(slopes, 2.5, atol=1e-9)


def test_ewma_features_run(prep, small_slice):
    out = prep.add_ewma_features(small_slice, spans=[5])
    ewma_cols = [c for c in out.columns if '_ewma_' in c]
    assert len(ewma_cols) == len(prep.sensor_cols)
    assert out[ewma_cols].notna().all().all()


def test_full_feature_pipeline_runs_end_to_end(prep, small_slice):
    """Smoke test for the whole chain, without touching the gold layer on disk."""
    df = prep.clip_rul(small_slice, clip_value=125)
    df = prep.add_rolling_features(df, windows=[5])
    df = prep.add_lag_features(df, lags=[1])
    df = prep.add_trend_features(df, window=5)
    df = prep.add_ewma_features(df, spans=[5])
    assert df.notna().all().all()
    assert np.isfinite(df.select_dtypes('number').to_numpy()).all()
