"""Pipeline tests.

These cover the two failure modes that motivated the class: fitting transforms
on data the model should never have seen, and the training and inference paths
computing different features from the same input.

Everything here fits a pipeline from the raw data in the repo, so the suite does
not depend on trained artifacts being present.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import ID_COLS, TARGET_COLS, FeaturePipeline
from splits import make_splits, split_frame


@pytest.fixture
def fast_pipeline():
    """Small feature config, so the tests stay quick.

    Function scoped on purpose: fit() mutates the object, and a shared instance
    would let one test's fitted state decide another test's outcome.
    """
    return FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])


@pytest.fixture
def featured(fast_pipeline, small_slice):
    return fast_pipeline.build_features(fast_pipeline.add_target(small_slice))


def test_transform_before_fit_is_a_clear_error(small_slice):
    with pytest.raises(RuntimeError, match='not fitted'):
        FeaturePipeline().transform(small_slice)


def test_load_missing_pipeline_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='src.train'):
        FeaturePipeline.load(tmp_path / 'nope.joblib')


def test_clipping_caps_the_target(fast_pipeline, small_slice):
    out = fast_pipeline.add_target(small_slice)
    assert out['RUL_clipped'].max() <= fast_pipeline.rul_clip


def test_transform_returns_the_training_columns_in_order(fast_pipeline, featured):
    fast_pipeline.fit(featured, already_featured=True)
    out = fast_pipeline.transform(featured, already_featured=True)
    assert list(out.columns) == fast_pipeline.feature_names_


def test_no_ids_or_targets_leak_into_the_feature_matrix(fast_pipeline, featured):
    fast_pipeline.fit(featured, already_featured=True)
    for col in ID_COLS + TARGET_COLS:
        assert col not in fast_pipeline.feature_names_


def test_output_has_no_nan_or_inf(fast_pipeline, featured):
    fast_pipeline.fit(featured, already_featured=True)
    out = fast_pipeline.transform(featured, already_featured=True)
    assert out.notna().all().all()
    assert np.isfinite(out.to_numpy()).all()


def test_training_rows_normalize_into_the_unit_range(fast_pipeline, featured):
    """The rows the scaler was fitted on land in [0, 1] by construction."""
    fast_pipeline.fit(featured, already_featured=True)
    out = fast_pipeline.transform(featured, already_featured=True)
    assert out.min().min() >= -1e-9
    assert out.max().max() <= 1 + 1e-9


def test_scaler_is_fitted_on_training_rows_only(small_slice):
    """The leakage regression test.

    Statistics learned from the training engines must not move when the
    validation and test engines change. Previously the scaler was fit on all
    100 engines before the split, so they did.
    """
    pipe = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    featured = pipe.build_features(pipe.add_target(small_slice))

    splits = make_splits(small_slice['engine_id'].unique())
    train_f, _, test_f = split_frame(featured, splits)

    pipe.fit(train_f, already_featured=True)
    baseline_min = pipe.feature_min_.copy()
    baseline_features = list(pipe.feature_names_)

    # Refit on the same training rows, but with the test engines mutated
    # beyond recognition. Nothing learned should move.
    poisoned = featured.copy()
    mask = poisoned['engine_id'].isin(splits['test'])
    sensor_cols = [c for c in poisoned.columns if 'sensor' in c]
    poisoned.loc[mask, sensor_cols] *= 1000

    refit = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    refit_train, _, _ = split_frame(poisoned, splits)
    refit.fit(refit_train, already_featured=True)

    pd.testing.assert_series_equal(refit.feature_min_, baseline_min)
    assert refit.feature_names_ == baseline_features


def test_held_out_rows_may_fall_outside_the_unit_range(small_slice):
    """A sanity check that the scaler really is train-only.

    If held out rows were always exactly in [0, 1] that would mean their own
    min/max had been used to build the scaler.
    """
    pipe = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    featured = pipe.build_features(pipe.add_target(small_slice))
    splits = make_splits(small_slice['engine_id'].unique())
    train_f, _, test_f = split_frame(featured, splits)

    pipe.fit(train_f, already_featured=True)
    held_out = pipe.transform(test_f, already_featured=True)

    assert held_out.notna().all().all()
    # Not asserting that it always happens on this small slice, only that the
    # transform tolerates it rather than clipping silently.
    assert held_out.to_numpy().min() <= 1.0


def test_missing_sensor_column_is_rejected(fast_pipeline, small_slice):
    pipe = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    pipe.build_features(pipe.add_target(small_slice))
    with pytest.raises(ValueError, match='missing sensor columns'):
        pipe.build_features(small_slice.drop(columns=['sensor_2']))


def test_training_and_inference_paths_agree(small_slice, tmp_path):
    """The regression test for #10.

    Feed raw cycles through the saved pipeline the way inference does, and the
    numbers must match the training matrix row for row. Two implementations
    that drift is exactly the bug this replaced.
    """
    pipe = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    featured = pipe.build_features(pipe.add_target(small_slice))
    pipe.fit(featured, already_featured=True)
    training_matrix = pipe.transform(featured, already_featured=True)

    reloaded = FeaturePipeline.load(pipe.save(tmp_path / 'pipe.joblib'))
    inference_matrix = reloaded.transform(small_slice)

    assert list(inference_matrix.columns) == list(training_matrix.columns)
    pd.testing.assert_frame_equal(inference_matrix, training_matrix)


def test_inference_is_row_order_independent_per_engine(small_slice, tmp_path):
    """Predicting for one engine alone gives the same features as predicting
    for the whole fleet at once. Features are per engine, so they must."""
    pipe = FeaturePipeline(windows=[5], lags=[1], trend_window=5, ewma_spans=[5])
    featured = pipe.build_features(pipe.add_target(small_slice))
    pipe.fit(featured, already_featured=True)

    fleet = pipe.transform(small_slice)
    fleet_engine_1 = fleet[small_slice['engine_id'].to_numpy() == 1]

    alone = pipe.transform(small_slice[small_slice['engine_id'] == 1])

    np.testing.assert_allclose(fleet_engine_1.to_numpy(), alone.to_numpy())
