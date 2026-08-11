"""Tests against the artifacts a real training run produces.

These are skipped when the artifacts are absent, which is the case in CI: the
workflow deliberately does not train, and models are build output rather than
source. Run `python -m src.train` locally and they become active.

The point of this file is the serving path. Everything else in the suite
verifies the training side in isolation; this checks that what gets loaded at
inference reproduces what the model was fitted on.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'data' / 'models' / 'cmapss'
SILVER = REPO_ROOT / 'data' / 'silver' / 'cmapss' / 'FD001_cleaned.csv'

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / 'feature_pipeline.joblib').exists(),
    reason='no trained artifacts; run `python -m src.train`')


@pytest.fixture(scope='module')
def metrics():
    with open(MODELS_DIR / 'metrics.json') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def cleaned():
    return pd.read_csv(SILVER)


def test_every_model_in_the_metadata_exists_on_disk():
    with open(MODELS_DIR / 'model_metadata.json') as f:
        metadata = json.load(f)
    for name, entry in metadata['models'].items():
        assert (MODELS_DIR / entry['file']).exists(), f'{name} artifact missing'


def test_sklearn_models_load_and_predict(cleaned):
    import joblib
    from pipeline import FeaturePipeline

    pipe = FeaturePipeline.load()
    sample = cleaned[cleaned['engine_id'].isin([1, 2])]
    X = pipe.transform(sample)

    for filename in ('gb_model.pkl', 'rf_model.pkl', 'ridge_model.pkl'):
        model = joblib.load(MODELS_DIR / filename)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert np.isfinite(preds).all()


def test_model_feature_count_matches_the_pipeline():
    """A mismatch here is the crash the dashboard used to hit on new data."""
    import joblib
    from pipeline import FeaturePipeline

    pipe = FeaturePipeline.load()
    for filename in ('gb_model.pkl', 'rf_model.pkl', 'ridge_model.pkl'):
        model = joblib.load(MODELS_DIR / filename)
        assert model.n_features_in_ == len(pipe.feature_names_)


def test_inference_path_reproduces_the_training_matrix(cleaned):
    """The regression test for the train/inference mismatch.

    Raw cycles pushed through the serving helper must give the same numbers as
    the pipeline applied directly. These were two separate implementations
    before, and they disagreed.
    """
    import feature_engineering as fe
    from pipeline import FeaturePipeline

    sample = cleaned[cleaned['engine_id'].isin([1, 2, 3])]

    direct = FeaturePipeline.load().transform(sample)
    served = fe.prepare_model_features(sample)

    assert list(served.columns) == list(direct.columns)
    pd.testing.assert_frame_equal(served, direct)


def test_lstm_sequence_has_the_shape_the_model_expects(cleaned):
    keras_file = MODELS_DIR / 'lstm_model.keras'
    if not keras_file.exists():
        pytest.skip('LSTM not trained')

    import feature_engineering as fe

    sample = cleaned[cleaned['engine_id'] == 1]
    seq = fe.prepare_lstm_sequence(sample)

    expected = (1, fe.get_lstm_sequence_length(), len(fe.get_lstm_feature_columns()))
    assert seq.shape == expected
    assert np.isfinite(seq).all()


def test_lstm_serving_applies_the_saved_scaler(cleaned):
    """Without the scaler the network sees a completely different scale.

    Standardised data is centred near zero; the min-max normalised features it
    used to be served sit in [0, 1]. Asserting the served window is centred is
    what catches the scaler silently going missing again.
    """
    if not (MODELS_DIR / 'lstm_scaler.joblib').exists():
        pytest.skip('LSTM not trained')

    import feature_engineering as fe

    sample = cleaned[cleaned['engine_id'] == 1]
    seq = fe.prepare_lstm_sequence(sample)

    assert seq.min() < 0, 'standardised inputs should include negative values'
    assert abs(float(seq.mean())) < 3


def test_metrics_report_both_validation_and_test(metrics):
    """Every model gets a test score. `test_r2: null` was the old state."""
    assert metrics['models'], 'no models in metrics.json'
    for name, m in metrics['models'].items():
        assert 'validation' in m and 'test' in m, name
        for split in ('validation', 'test'):
            assert m[split]['r2'] is not None, f'{name} has no {split} r2'
            assert np.isfinite(m[split]['r2'])


def test_test_scores_are_not_better_than_validation_by_a_lot(metrics):
    """Sanity check on the split.

    Test beating validation by a wide margin usually means the split leaked or
    the two were computed on different data, which is exactly what happened
    when the notebooks disagreed about which engines were held out.
    """
    for name, m in metrics['models'].items():
        gap = m['test']['r2'] - m['validation']['r2']
        assert gap < 0.10, f'{name} scores {gap:.3f} higher on test than validation'


def test_metadata_and_metrics_agree(metrics):
    """One source of truth. These used to be maintained separately and drift."""
    with open(MODELS_DIR / 'model_metadata.json') as f:
        metadata = json.load(f)

    for name, entry in metadata['models'].items():
        assert name in metrics['models'], f'{name} in metadata but not metrics'
        assert entry['metrics']['test_r2'] == pytest.approx(
            metrics['models'][name]['test']['r2'], abs=1e-4)
