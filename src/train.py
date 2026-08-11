"""Single reproducible path from raw data to trained models and metrics.

    python -m src.train              # everything
    python -m src.train --skip-lstm  # skip the slow one

The notebooks are the narrative: they show the exploration, the plots and the
reasoning. This script is the reproducible artifact. Previously the only way to
regenerate the models was to open six notebooks and run them in the right order,
which meant a fresh clone could not produce them at all.

Order matters here and is the point of the whole file:

1. clean the raw data (silver)
2. split by engine, and write the split down
3. engineer features, which is per engine and therefore split safe
4. fit the scaler and the correlation filter on TRAIN ROWS ONLY
5. tune on validation
6. score once on test, at the end
7. fit the survival models on a landmark design, scored on held out engines
8. write one metrics.json that every other surface reads
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import CMAPSSLoader
from pipeline import ID_COLS, TARGET_COLS, FeaturePipeline
from preprocessor import CMAPSSPreprocessor
from splits import make_splits, save_splits, split_frame

BASE_DIR = Path(__file__).resolve().parent.parent
BRONZE_DIR = BASE_DIR / 'data' / 'bronze' / 'cmapss'
SILVER_DIR = BASE_DIR / 'data' / 'silver' / 'cmapss'
GOLD_DIR = BASE_DIR / 'data' / 'gold' / 'cmapss'
MODELS_DIR = BASE_DIR / 'data' / 'models' / 'cmapss'

METRICS_FILE = MODELS_DIR / 'metrics.json'
METADATA_FILE = MODELS_DIR / 'model_metadata.json'
LSTM_SCALER_FILE = MODELS_DIR / 'lstm_scaler.joblib'
LSTM_FEATURES_FILE = MODELS_DIR / 'lstm_features.json'

DATASET = 'FD001'
SEED = 42
SEQUENCE_LENGTH = 30
TOP_N_FEATURES = 20


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def nasa_score(y_true, y_pred) -> float:
    """The asymmetric CMAPSS scoring function. Lower is better.

    Late predictions (the model thinks there is more life left than there is)
    are penalised harder than early ones, because in maintenance an optimistic
    error is the one that strands an aircraft.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(y_true, y_pred) -> dict:
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
        'nasa_score': nasa_score(y_true, y_pred),
        'n': int(len(y_true)),
    }


def residual_quantiles(y_true, y_pred) -> dict:
    """Empirical error distribution, used for honest prediction intervals.

    The dashboard used to draw its 95 percent band from random noise scaled off
    the prediction itself. These are the actual held out residuals, so an
    interval built from them means something.
    """
    resid = np.asarray(y_pred) - np.asarray(y_true)
    return {
        'p2.5': float(np.percentile(resid, 2.5)),
        'p50': float(np.percentile(resid, 50)),
        'p97.5': float(np.percentile(resid, 97.5)),
        'std': float(np.std(resid)),
    }


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def load_and_clean() -> pd.DataFrame:
    print('[1/8] loading and cleaning raw data')
    loader = CMAPSSLoader(BRONZE_DIR)
    raw = loader.load_dataset(DATASET, split='train')

    prep = CMAPSSPreprocessor(raw)
    cleaned, dropped = prep.clean_data(raw)
    print(f'      dropped sensors: constant={len(dropped["constant"])} '
          f'low_variance={len(dropped["low_variance"])} '
          f'correlated={len(dropped["correlated"])}')

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(SILVER_DIR / f'{DATASET}_cleaned.csv', index=False)
    print(f'      silver: {cleaned.shape}')
    return cleaned


def build_matrices(cleaned: pd.DataFrame):
    """Split first, then fit the pipeline on training rows only."""
    print('[2/8] splitting by engine')
    splits = make_splits(cleaned['engine_id'].unique(), seed=SEED)
    save_splits(splits)
    print(f'      train={len(splits["train"])} val={len(splits["val"])} '
          f'test={len(splits["test"])} engines')

    print('[3/8] engineering features')
    pipe = FeaturePipeline()
    targeted = pipe.add_target(cleaned)
    featured = pipe.build_features(targeted)
    train_f, val_f, test_f = split_frame(featured, splits)

    print('[4/8] fitting scaler and correlation filter on training engines only')
    pipe.fit(train_f, already_featured=True)
    print(f'      kept {len(pipe.feature_names_)} features, '
          f'dropped {len(pipe.dropped_correlated_)} correlated')
    pipe.save()

    def xy(frame):
        return (pipe.transform(frame, already_featured=True),
                frame['RUL_clipped'].reset_index(drop=True))

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold = pd.concat(
        [featured[ID_COLS + TARGET_COLS].reset_index(drop=True),
         pipe.transform(featured, already_featured=True).reset_index(drop=True)],
        axis=1)
    gold.to_csv(GOLD_DIR / f'{DATASET}_featured.csv', index=False)
    print(f'      gold: {gold.shape}')

    return pipe, splits, xy(train_f), xy(val_f), xy(test_f), featured


def train_sklearn_models(train, val, test):
    print('[5/8] training scikit-learn models')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test
    models, metrics = {}, {}

    def record(name, model, tuning=None):
        models[name] = model
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        metrics[name] = {
            'validation': evaluate(y_val, val_pred),
            'test': evaluate(y_test, test_pred),
            'residuals_test': residual_quantiles(y_test, test_pred),
        }
        if tuning:
            metrics[name]['tuning'] = tuning
        print(f'      {name:<18} val r2={metrics[name]["validation"]["r2"]:.4f} '
              f'test r2={metrics[name]["test"]["r2"]:.4f} '
              f'test mae={metrics[name]["test"]["mae"]:.2f}')

    record('linear_regression', LinearRegression().fit(X_train, y_train))

    # Alphas are picked on validation. Test is not consulted here, which is the
    # whole reason there are three splits and not two.
    for name, factory in (('ridge', lambda a: Ridge(alpha=a)),
                          ('lasso', lambda a: Lasso(alpha=a, max_iter=10000))):
        scored = []
        for alpha in (0.1, 1.0, 10.0, 100.0):
            candidate = factory(alpha).fit(X_train, y_train)
            rmse = float(np.sqrt(mean_squared_error(y_val, candidate.predict(X_val))))
            scored.append((rmse, alpha))
        best_rmse, best_alpha = min(scored)
        record(name, factory(best_alpha).fit(X_train, y_train),
               tuning={'selected_alpha': best_alpha,
                       'searched': {str(a): r for r, a in scored}})

    record('random_forest', RandomForestRegressor(
        n_estimators=150, max_depth=12, min_samples_split=15, min_samples_leaf=6,
        max_features='sqrt', random_state=SEED, n_jobs=-1).fit(X_train, y_train))

    record('gradient_boosting', GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4, min_samples_split=15,
        min_samples_leaf=6, subsample=0.8, max_features='sqrt',
        random_state=SEED).fit(X_train, y_train))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {'linear_regression': 'lr_model.pkl', 'ridge': 'ridge_model.pkl',
                 'lasso': 'lasso_model.pkl', 'random_forest': 'rf_model.pkl',
                 'gradient_boosting': 'gb_model.pkl'}
    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / filenames[name])

    return models, metrics, filenames


def select_top_features(models, feature_names) -> list:
    """Consensus importance from the two tree models, fitted on train only."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'rf': models['random_forest'].feature_importances_,
        'gb': models['gradient_boosting'].feature_importances_,
    })
    importance['mean'] = importance[['rf', 'gb']].mean(axis=1)
    top = importance.sort_values('mean', ascending=False).head(TOP_N_FEATURES)
    return top['feature'].tolist()


def make_sequences(frame: pd.DataFrame, cols: list, length: int):
    """Sliding windows within each engine. Windows never span two engines."""
    xs, ys = [], []
    for _, engine in frame.groupby('engine_id', sort=True):
        engine = engine.sort_values('time_cycles')
        values = engine[cols].to_numpy()
        targets = engine['RUL_clipped'].to_numpy()
        for i in range(len(engine) - length + 1):
            xs.append(values[i:i + length])
            ys.append(targets[i + length - 1])
    return np.asarray(xs), np.asarray(ys)


def train_lstm(pipe, splits, featured, top_features):
    print('[6/8] training LSTM')
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras

    keras.utils.set_random_seed(SEED)

    normalized = pipe.transform(featured, already_featured=True)
    frame = pd.concat([featured[ID_COLS + TARGET_COLS].reset_index(drop=True),
                       normalized.reset_index(drop=True)], axis=1)
    train_f, val_f, test_f = split_frame(frame, splits)

    X_train, y_train = make_sequences(train_f, top_features, SEQUENCE_LENGTH)
    X_val, y_val = make_sequences(val_f, top_features, SEQUENCE_LENGTH)
    X_test, y_test = make_sequences(test_f, top_features, SEQUENCE_LENGTH)
    print(f'      sequences train={X_train.shape} val={X_val.shape} test={X_test.shape}')

    # Fit on training sequences only, and unlike before, actually keep it.
    n_features = len(top_features)
    scaler = StandardScaler().fit(X_train.reshape(-1, n_features))

    def scale(x):
        return scaler.transform(x.reshape(-1, n_features)).reshape(x.shape)

    model = keras.Sequential([
        keras.layers.Input(shape=(SEQUENCE_LENGTH, n_features)),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='mae', metrics=['mae'])

    model.fit(
        scale(X_train), y_train,
        validation_data=(scale(X_val), y_val),
        epochs=150, batch_size=128, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=25,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                              patience=10, min_lr=1e-6),
        ])

    val_pred = model.predict(scale(X_val), verbose=0).flatten()
    test_pred = model.predict(scale(X_test), verbose=0).flatten()
    metrics = {
        'validation': evaluate(y_val, val_pred),
        'test': evaluate(y_test, test_pred),
        'residuals_test': residual_quantiles(y_test, test_pred),
    }
    print(f'      {"lstm":<18} val r2={metrics["validation"]["r2"]:.4f} '
          f'test r2={metrics["test"]["r2"]:.4f} '
          f'test mae={metrics["test"]["mae"]:.2f}')

    model.save(MODELS_DIR / 'lstm_model.keras')
    joblib.dump(scaler, LSTM_SCALER_FILE)
    with open(LSTM_FEATURES_FILE, 'w') as f:
        json.dump({'features': top_features, 'sequence_length': SEQUENCE_LENGTH}, f, indent=2)

    return metrics


def train_survival(cleaned, splits):
    """Fit the survival models on the landmark design and score held out engines."""
    print('[7/8] training survival models')
    import survival

    frame = survival.build_landmark_frame(cleaned)
    train_f = frame[frame['engine_id'].isin(splits['train'])]
    val_f = frame[frame['engine_id'].isin(splits['val'])]
    test_f = frame[frame['engine_id'].isin(splits['test'])]

    covariates = survival.select_covariates(train_f)
    models, scaler, design = survival.fit_survival_models(train_f, covariates)
    print(f'      landmark={survival.LANDMARK} horizon={survival.HORIZON} '
          f'engines={len(frame)} censored={int((frame["event"] == 0).sum())}')

    metrics = {}
    for name, model in models.items():
        metrics[name] = {
            'validation': {'concordance': survival.concordance(model, design(val_f))},
            'test': {'concordance': survival.concordance(model, design(test_f))},
            'train_concordance': survival.concordance(model, design(train_f)),
        }
        print(f'      {name:<18} val c={metrics[name]["validation"]["concordance"]:.3f} '
              f'test c={metrics[name]["test"]["concordance"]:.3f}')

    joblib.dump(models['weibull'], MODELS_DIR / 'waft.pkl')
    joblib.dump(models['cox'], MODELS_DIR / 'cph.pkl')
    joblib.dump({'scaler': scaler, 'covariates': covariates,
                 'landmark': survival.LANDMARK, 'window': survival.WINDOW,
                 'horizon': survival.HORIZON},
                MODELS_DIR / 'survival_design.joblib')
    return metrics


def write_artifacts(pipe, splits, metrics, filenames, cleaned, elapsed, lstm_trained,
                    survival_metrics=None):
    print('[8/8] writing metrics and metadata')
    payload = {
        'generated': date.today().isoformat(),
        'dataset': DATASET,
        'seed': SEED,
        'split': {name: len(splits[name]) for name in ('train', 'val', 'test')},
        'n_features': len(pipe.feature_names_),
        'training_seconds': round(elapsed, 1),
        'models': metrics,
        'survival': survival_metrics or {},
        'notes': (
            'Validation is used for hyperparameter selection. Test is scored '
            'once, after selection, and is what should be reported. Survival '
            'concordance is reported on held out engines, not the training '
            'concordance_index_ attribute.'),
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)

    descriptions = {
        'linear_regression': 'Unregularized linear baseline',
        'ridge': 'Linear baseline with L2 regularization',
        'lasso': 'Linear baseline with L1 regularization and feature selection',
        'random_forest': 'Random forest, 150 trees, max_depth 12',
        'gradient_boosting': 'Gradient boosting, 150 estimators, learning_rate 0.05',
        'lstm': f'LSTM, 32 units, sequence length {SEQUENCE_LENGTH}',
    }
    models_meta = {}
    for name, m in metrics.items():
        models_meta[name] = {
            'file': filenames.get(name, 'lstm_model.keras'),
            'framework': 'tensorflow' if name == 'lstm' else 'scikit-learn',
            'features': (TOP_N_FEATURES if name == 'lstm'
                         else len(pipe.feature_names_)),
            'trained_date': date.today().isoformat(),
            'description': descriptions.get(name, ''),
            'metrics': {
                'validation_r2': round(m['validation']['r2'], 4),
                'validation_mae': round(m['validation']['mae'], 2),
                'test_r2': round(m['test']['r2'], 4),
                'test_mae': round(m['test']['mae'], 2),
            },
        }

    metadata = {
        'models': models_meta,
        'dataset': {
            'name': f'CMAPSS {DATASET}',
            'engines': int(cleaned['engine_id'].nunique()),
            'total_records': int(len(cleaned)),
            'sensors': len(pipe.sensor_cols_),
            'rul_clip_value': pipe.rul_clip,
        },
        'last_updated': date.today().isoformat(),
        'lstm_trained': lstm_trained,
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'      {METRICS_FILE.relative_to(BASE_DIR)}')
    print(f'      {METADATA_FILE.relative_to(BASE_DIR)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-lstm', action='store_true',
                        help='skip the LSTM, which dominates the runtime')
    args = parser.parse_args()

    started = time.perf_counter()
    cleaned = load_and_clean()
    pipe, splits, train, val, test, featured = build_matrices(cleaned)
    models, metrics, filenames = train_sklearn_models(train, val, test)

    top_features = select_top_features(models, pipe.feature_names_)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([featured[ID_COLS + TARGET_COLS].reset_index(drop=True),
               pipe.transform(featured, already_featured=True)[top_features]
                   .reset_index(drop=True)], axis=1) \
      .to_csv(GOLD_DIR / f'{DATASET}_top_features.csv', index=False)

    if args.skip_lstm:
        print('[6/8] skipping LSTM')
    else:
        metrics['lstm'] = train_lstm(pipe, splits, featured, top_features)

    survival_metrics = train_survival(cleaned, splits)

    elapsed = time.perf_counter() - started
    write_artifacts(pipe, splits, metrics, filenames, cleaned, elapsed,
                    lstm_trained=not args.skip_lstm,
                    survival_metrics=survival_metrics)
    print(f'\ndone in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
