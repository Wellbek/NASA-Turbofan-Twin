import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow import keras
import shap
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = DATA_DIR / 'models' / 'cmapss'
SILVER_DIR = DATA_DIR / 'silver' / 'cmapss'
GOLD_DIR = DATA_DIR / 'gold' / 'cmapss'
METADATA_FILE = MODELS_DIR / 'model_metadata.json'
METRICS_FILE = MODELS_DIR / 'metrics.json'
SPLITS_FILE = MODELS_DIR / 'splits.json'

# The models were trained on engineered, normalized features, so inference has
# to rebuild the exact same feature space. src/feature_engineering.py does that
# by loading the pipeline that training fitted and saved, rather than
# reimplementing the formulas here.
sys.path.insert(0, str(BASE_DIR / 'src'))
import feature_engineering as fe

@st.cache_data
def _lstm_feature_columns():
    """The LSTM feature columns (cached, read once)."""
    return fe.get_lstm_feature_columns()

@st.cache_data
def _model_feature_columns():
    """The tree/linear model feature columns (cached, read once)."""
    return fe.get_model_feature_columns()


# --------------------------------------------------------------------------
# Metrics
#
# Every number this dashboard shows comes from the artifact the training run
# writes. They used to be typed into about ten places by hand and had already
# drifted out of agreement with each other, with model_metadata.json and with
# the README.
# --------------------------------------------------------------------------

DISPLAY_NAMES = {
    'lstm': 'LSTM',
    'gradient_boosting': 'Gradient Boosting',
    'random_forest': 'Random Forest',
    'ridge': 'Ridge',
    'lasso': 'Lasso',
    'linear_regression': 'Linear Regression',
    'weibull': 'Weibull AFT',
    'cox': 'Cox PH',
}

MODEL_ROLES = {
    'lstm': 'Sequence model, needs 30 cycles of history',
    'gradient_boosting': 'Strong tabular baseline, fast and explainable',
    'random_forest': 'Interpretable ensemble, feature importance',
    'ridge': 'Regularized linear baseline',
    'lasso': 'Sparse linear baseline',
    'linear_regression': 'Unregularized reference point',
}


@st.cache_data
def load_metrics():
    """Validation and test scores written by `python -m src.train`."""
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE) as f:
        return json.load(f)


def display_name(key):
    return DISPLAY_NAMES.get(key, key.replace('_', ' ').title())


def metrics_table(metrics, split='test'):
    """Long form table of every regression model's scores for one split."""
    rows = [{
        'key': key,
        'Model': display_name(key),
        'R2': m[split]['r2'],
        'MAE': m[split]['mae'],
        'RMSE': m[split]['rmse'],
        'NASA score': m[split]['nasa_score'],
        'Role': MODEL_ROLES.get(key, ''),
    } for key, m in metrics['models'].items()]
    return pd.DataFrame(rows).sort_values('R2', ascending=False).reset_index(drop=True)


def best_model_key(metrics, split='test'):
    return max(metrics['models'].items(), key=lambda kv: kv[1][split]['r2'])[0]


def require_metrics():
    """Show a clear message instead of inventing numbers when untrained."""
    metrics = load_metrics()
    if metrics is None:
        st.warning(
            "No metrics artifact found. Run `python -m src.train` from the repo "
            "root to train the models and generate data/models/cmapss/metrics.json."
        )
    return metrics

# Page configuration
st.set_page_config(
    page_title="Turbofan Engine Predictive Maintenance",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.25rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
    }
    .critical-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    /* Improve sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        font-weight: 600;
        color: #333;
    }
    /* Improve main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and metadata
@st.cache_resource
def load_models_with_metadata():
    """Load all trained models and their metadata"""
    try:
        # Load metadata
        if not METADATA_FILE.exists():
            st.warning("Model metadata file not found. Loading models without validation.")
            return load_models_basic()

        import json
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        models = {}
        models_dir = MODELS_DIR

        if not models_dir.exists():
            st.error(f"Models directory not found: {models_dir}")
            return None

        # Load each model from metadata
        for model_name, model_info in metadata['models'].items():
            model_file = models_dir / model_info['file']

            if not model_file.exists():
                st.warning(f"Model file not found: {model_file}")
                continue

            try:
                if model_info['framework'] == 'tensorflow':
                    models[model_name] = keras.models.load_model(model_file)
                elif model_info['framework'] == 'lifelines':
                    models[model_name] = joblib.load(model_file)
                else:
                    models[model_name] = joblib.load(model_file)

                # Add metadata to model object
                models[model_name].metadata = model_info

            except Exception as e:
                st.error(f"Failed to load {model_name}: {e}")

        # Survival models are not in the metrics-shaped metadata block, since
        # they are scored by concordance rather than R2, so they are loaded by
        # name. Without this the dashboard silently has no survival model and
        # every survival section renders empty.
        for name, filename in (('weibull', 'waft.pkl'), ('cox', 'cph.pkl')):
            path = models_dir / filename
            if path.exists():
                try:
                    models[name] = joblib.load(path)
                except Exception as e:
                    st.warning(f"Could not load {name}: {e}")

        # Gradient Boosting must load natively. Never silently substitute another
        # model - a missing model should be visible, not disguised as a different one.
        if 'gradient_boosting' not in models:
            st.error(
                "Gradient Boosting model could not be loaded. "
                "Run `python -m src.train` from the repo root to regenerate "
                "data/models/cmapss/gb_model.pkl."
            )

        return models, metadata

    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return load_models_basic()

def load_models_basic():
    """Fallback function to load models without metadata"""
    try:
        models_dir = MODELS_DIR

        if not models_dir.exists():
            st.error(f"Models directory not found: {models_dir}")
            return None

        models = {}

        # Load traditional ML models with fallback
        try:
            models['ridge'] = joblib.load(models_dir / 'ridge_model.pkl')
        except Exception as e:
            st.warning(f"Could not load Ridge model: {e}")

        try:
            models['random_forest'] = joblib.load(models_dir / 'rf_model.pkl')
        except Exception as e:
            st.warning(f"Could not load Random Forest model: {e}")

        try:
            models['gradient_boosting'] = joblib.load(models_dir / 'gb_model.pkl')
        except Exception as e:
            st.error(f"Could not load Gradient Boosting model: {e}")

        # Load survival models
        try:
            models['weibull'] = joblib.load(models_dir / 'waft.pkl')
        except Exception as e:
            st.warning(f"Could not load Weibull model: {e}")

        try:
            models['cox'] = joblib.load(models_dir / 'cph.pkl')
        except Exception as e:
            st.warning(f"Could not load Cox model: {e}")

        # Load LSTM model
        try:
            models['lstm'] = keras.models.load_model(models_dir / 'lstm_model.keras')
        except Exception as e:
            st.warning(f"Could not load LSTM model: {e}")

        # Gradient Boosting must load natively. Never silently substitute another
        # model - a missing model should be visible, not disguised as a different one.
        if 'gradient_boosting' not in models:
            st.error(
                "Gradient Boosting model could not be loaded. "
                "Run `python -m src.train` from the repo root to regenerate "
                "data/models/cmapss/gb_model.pkl."
            )

        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_splits():
    """Which engines were train, validation and test."""
    if not SPLITS_FILE.exists():
        return None
    with open(SPLITS_FILE) as f:
        return json.load(f)


@st.cache_data
def load_evaluation_data(split='test'):
    """Engineered features for one split, defaulting to held out engines.

    This function was previously called `load_test_data`, and it loaded the
    entire featured file: all 100 engines, roughly 70 of which the models were
    fitted on. Engine Analysis and Fleet Management both ran on it, so most of
    what the dashboard scored and risk ranked was training data being presented
    as held out.

    Now it filters to the split recorded in splits.json, and the pages say which
    split they are showing.
    """
    try:
        df = pd.read_csv(GOLD_DIR / 'FD001_featured.csv')
        feature_cols = _model_feature_columns()
        keep = ['engine_id', 'time_cycles'] + [c for c in feature_cols if c in df.columns]
        df = df[keep]

        splits = load_splits()
        if splits is None or split == 'all':
            return df
        return df[df['engine_id'].isin(splits[split])].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return None


@st.cache_data
def load_raw_cycles():
    """Cleaned raw sensor cycles, needed by the survival covariate builder."""
    path = SILVER_DIR / 'FD001_cleaned.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_lstm_scaler():
    """The StandardScaler the LSTM was trained with."""
    path = MODELS_DIR / 'lstm_scaler.joblib'
    if not path.exists():
        return None
    return joblib.load(path)


def build_lstm_window(engine_data):
    """Last N cycles of the LSTM feature subset, scaled the way it was trained.

    The pipeline's min-max normalization is only half of what the LSTM expects.
    It was also fitted on standardised sequences, so the saved StandardScaler
    has to be applied here too. Leaving it out feeds the network inputs on a
    scale it never saw, which is the train/serve skew this project already had
    once when the scaler was not saved at all.
    """
    length = fe.get_lstm_sequence_length()
    lstm_cols = [c for c in _lstm_feature_columns() if c in engine_data.columns]
    seq = engine_data[lstm_cols].iloc[-length:].to_numpy()

    if 0 < len(seq) < length:
        pad = np.repeat(seq[:1], length - len(seq), axis=0)
        seq = np.vstack([pad, seq])
    elif len(seq) == 0:
        seq = np.zeros((length, len(lstm_cols)))

    scaler = load_lstm_scaler()
    if scaler is not None:
        seq = scaler.transform(seq)
    return seq.reshape(1, length, len(lstm_cols))


# Generate predictions
def predict_rul(engine_data, models, model_type='lstm'):
    """Generate RUL predictions for an engine.

    ``engine_data`` is a DataFrame of the model's engineered, normalized
    features. LSTM uses its own column subset over the last 30 cycles;
    tree/linear models use the last cycle's full feature row.
    """
    try:
        if model_type == 'lstm':
            X = build_lstm_window(engine_data)
            prediction = models['lstm'].predict(X, verbose=0)[0][0]
        else:
            X = engine_data.iloc[-1:].values
            prediction = models[model_type].predict(X)[0]
        return max(0, float(prediction))
    except Exception as e:
        st.error(f"Error predicting RUL with {model_type}: {e}")
        return None

@st.cache_resource
def load_survival_design():
    """The standardiser and covariate list the survival models were fit with."""
    path = MODELS_DIR / 'survival_design.joblib'
    if not path.exists():
        return None
    return joblib.load(path)


def calculate_survival_probability(engine_raw, models, time_horizons=(25, 50, 75, 100)):
    """Survival probabilities from the fitted Weibull AFT model.

    The previous version hardcoded a Weibull shape of 2.5 and a scale of
    `rul * 1.2`, under a comment claiming the parameters were fitted from
    training data. Nothing was fitted. Meanwhile waft.pkl sat on disk, loaded by
    this dashboard and never used for anything.

    This builds the same landmark covariates the model was trained on and asks
    it. Returns None when the model or its inputs are unavailable, so the caller
    can hide the section instead of drawing an invented curve.
    """
    design = load_survival_design()
    model = models.get('weibull')
    if design is None or model is None or engine_raw is None:
        return None

    try:
        import survival

        frame = survival.build_landmark_frame(
            engine_raw,
            landmark=design['landmark'],
            window=design['window'],
            horizon=design['horizon'],
        )
        if frame.empty:
            # Engine has not run long enough to reach the landmark, so the
            # survival model has nothing to say about it yet.
            return None

        covariates = design['covariates']
        standardized = pd.DataFrame(
            design['scaler'].transform(frame[covariates]),
            columns=covariates, index=frame.index)

        curve = model.predict_survival_function(standardized.iloc[[0]],
                                                times=list(time_horizons))
        return {int(t): float(curve.loc[t].iloc[0]) for t in time_horizons}
    except Exception:
        return None

# Risk classification
def classify_risk(rul):
    """Classify engine risk level based on RUL"""
    if rul < 30:
        return "CRITICAL", "critical"
    elif rul < 60:
        return "WARNING", "warning"
    else:
        return "HEALTHY", "success"

# Process uploaded CSV data
def process_uploaded_csv(uploaded_file):
    """Process uploaded CSV file and return formatted raw sensor data.

    Required columns: engine_id, time_cycles, and the 13 FD001 sensors.
    Operational settings are optional - if absent they are filled with FD001
    cruise defaults, since they are part of the model's feature space.
    """
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ['engine_id', 'time_cycles'] + fe.RAW_SENSORS
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None

        keep = ['engine_id', 'time_cycles'] + fe.OPERATIONAL_SETTINGS + fe.RAW_SENSORS
        df = df[[c for c in keep if c in df.columns]].copy()

        # Default operational settings to FD001 cruise values if not provided.
        for col in fe.OPERATIONAL_SETTINGS:
            if col not in df.columns:
                df[col] = 0.0 if col != 'operational_setting_3' else 100.0

        # Check if we have enough cycles for LSTM (need at least 30)
        unique_engines = df['engine_id'].nunique()
        total_cycles = len(df)

        return df, {
            'unique_engines': unique_engines,
            'total_cycles': total_cycles,
            'can_use_lstm': total_cycles >= 30
        }
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None

def engineer_for_prediction(raw_df):
    """Engineer raw sensor data into the 153 normalized model features.

    Returns a DataFrame with the model's training columns (same row order as
    the input) plus an ``engine_id`` column for grouping by engine.
    """
    features = fe.prepare_model_features(raw_df).copy()
    features['engine_id'] = raw_df['engine_id'].values
    return features

# Generate prediction for new data
def generate_new_prediction(raw_df, models, model_type='gradient_boosting'):
    """Generate RUL predictions for new raw sensor data.

    Engineers the raw upload into the model's 153-feature space, then predicts
    using the most recent cycle of each engine (tree/linear models) or the last
    30-cycle sequence (LSTM).
    """
    try:
        features = engineer_for_prediction(raw_df)
        model_cols = _model_feature_columns()
        feature_df = features[model_cols]

        if model_type == 'lstm':
            prediction = predict_rul(feature_df, models, 'lstm')
        else:
            # Most recent cycle across the uploaded engines
            latest_idx = features.groupby('engine_id').tail(1).index
            X = features.loc[latest_idx, model_cols].values
            prediction = float(models[model_type].predict(X)[0])

        if prediction is None:
            return None, None
        return max(0, prediction), features.groupby('engine_id').tail(1).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return None, None



def ensemble_weights(metrics, keys):
    """Weights proportional to held out R2, computed rather than hardcoded.

    The old weights were frozen at 0.45 / 0.35 / 0.20 with a docstring
    justifying them from numbers that have since changed. Deriving them means a
    retrain updates the ensemble instead of silently invalidating it.

    R2 is shifted by a floor below the weakest candidate so a model that is only
    marginally better does not take a wildly larger share.
    """
    available = {k: metrics['models'][k]['test']['r2']
                 for k in keys if k in metrics.get('models', {})}
    if not available:
        return {}

    floor = min(available.values()) - 0.05
    raw = {k: v - floor for k, v in available.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def predict_ensemble_rul(engine_data, models, metrics=None):
    """Weighted average of the models that can score this input.

    Weights come from held out R2 in the metrics artifact. Models that are not
    loaded are dropped and the remaining weights renormalise, so a missing LSTM
    degrades the ensemble instead of silently contributing a zero.
    """
    predictions = {}

    # LSTM prediction over its own feature subset and sequence length.
    if 'lstm' in models and len(engine_data) >= 1:
        X_lstm = build_lstm_window(engine_data)
        predictions['lstm'] = float(models['lstm'].predict(X_lstm, verbose=0)[0][0])

    # Gradient Boosting prediction (use last cycle, full feature set)
    if 'gradient_boosting' in models and len(engine_data) >= 1:
        X_gb = engine_data.iloc[-1:].values
        predictions['gradient_boosting'] = float(models['gradient_boosting'].predict(X_gb)[0])

    # Random Forest prediction (use last cycle, full feature set)
    if 'random_forest' in models and len(engine_data) >= 1:
        X_rf = engine_data.iloc[-1:].values
        predictions['random_forest'] = float(models['random_forest'].predict(X_rf)[0])

    if not predictions or metrics is None:
        return None

    weights = ensemble_weights(metrics, list(predictions))
    if not weights:
        return None

    ensemble_pred = sum(predictions[k] * w for k, w in weights.items())
    return max(0, ensemble_pred), predictions, weights

def get_prediction_interval(rul_pred, model_key, metrics):
    """95% prediction interval from the model's own held out residuals.

    The previous implementation drew random noise scaled off the prediction
    itself, then used the 2.5th percentile for *both* bounds, producing a narrow
    symmetric band that encoded nothing at all. It was labelled "95% Confidence"
    in the UI.

    These quantiles are the actual test set errors recorded during training, so
    the width reflects how wrong this model has really been on engines it had
    not seen. Returns None when there is nothing to base an interval on, and
    callers hide the section rather than substituting a made up one.
    """
    if metrics is None:
        return None

    entry = metrics.get('models', {}).get(model_key)
    if not entry or 'residuals_test' not in entry:
        return None

    residuals = entry['residuals_test']
    # residual = prediction - truth, so subtract to recover the plausible
    # range for the truth given this prediction.
    lower = rul_pred - residuals['p97.5']
    upper = rul_pred - residuals['p2.5']
    return max(0.0, lower), max(0.0, upper)


# Main dashboard# SHAP explainability
def get_shap_explanations(engine_data, models, model_type='gradient_boosting', n_features=5):
    """Get SHAP values for feature importance explanation

    Args:
    engine_data: Engine sensor data (DataFrame)
    models: Dictionary of loaded models
    model_type: Which model to explain
    n_features: Number of top features to return

    Returns:
    Dictionary with feature names and importance values
    """
    try:
        if model_type not in models:
            return None

        model = models[model_type]

        # Use last cycle for explanation
        if hasattr(engine_data, 'iloc'):
            X = engine_data.iloc[-1:].values
            feature_names = list(engine_data.columns)
        else:
            X = engine_data.values[-1:].reshape(1, -1)
            feature_names = _model_feature_columns()

        # Compute SHAP values (use TreeExplainer for tree-based models)
        if model_type in ['random_forest', 'gradient_boosting']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)[0]
        else:
            # Use KernelExplainer as fallback
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)[0]

        # Get feature importance
        feature_importance = {}
        for name, value in zip(feature_names, shap_values):
            feature_importance[name] = abs(value)

        # Sort and return top N features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:n_features])

    except Exception as e:
        st.error(f"Error computing SHAP values: {e}")
        return None



# --------------------------------------------------------------------------
# Shared rendering
#
# These three blocks were duplicated across three pages with slightly different
# wording and, in the interval case, slightly different maths. One copy each.
# --------------------------------------------------------------------------

def render_prediction_interval(rul, interval, model_key):
    """Show the 95% interval, or explain why there isn't one."""
    st.markdown("---")
    st.subheader("Prediction Interval (95%)")

    if interval is None:
        st.info(
            "No interval available. Intervals come from the model's held out "
            "residuals in metrics.json, so they need a completed training run."
        )
        return

    lower, upper = interval
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower bound", f"{lower:.1f} cycles")
    with col2:
        st.metric("Point estimate", f"{rul:.1f} cycles")
    with col3:
        st.metric("Upper bound", f"{upper:.1f} cycles")

    st.caption(
        f"Width {upper - lower:.1f} cycles, from the 2.5th and 97.5th percentile "
        f"of {display_name(model_key)}'s errors on held out engines. The band is "
        "asymmetric because the model's errors are."
    )


def render_individual_predictions(individual_preds, weights):
    """Per-model contributions to an ensemble prediction."""
    if not individual_preds or not weights:
        return

    st.markdown("---")
    st.subheader("Individual Model Predictions")

    columns = st.columns(len(individual_preds))
    for col, (key, value) in zip(columns, individual_preds.items()):
        with col:
            st.metric(f"{display_name(key)} ({weights.get(key, 0):.0%})",
                      f"{value:.1f} cycles")

    st.caption(
        "Weights are proportional to held out R2, recomputed from metrics.json "
        "on each load rather than fixed in the code."
    )


def render_survival_section(engine_raw, models, horizons=(10, 25, 50, 75, 100, 125)):
    """Survival curve from the fitted Weibull model, or nothing."""
    probabilities = calculate_survival_probability(engine_raw, models, horizons)

    st.markdown("---")
    st.subheader("Survival Probability")

    if probabilities is None:
        st.info(
            "No survival curve for this engine. The Weibull model is fitted on a "
            "landmark design and needs at least "
            f"{load_survival_design()['landmark'] if load_survival_design() else 100} "
            "cycles of history before it can say anything."
        )
        return

    times = sorted(probabilities)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=[probabilities[t] for t in times],
        mode='lines+markers',
        name='Survival probability',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        fill='tozeroy',
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                  annotation_text="50% survival")
    fig.update_layout(
        title="Probability of surviving N more cycles past the landmark",
        xaxis_title="Cycles",
        yaxis_title="Survival probability",
        yaxis_range=[0, 1],
        hovermode='x unified',
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "From the fitted Weibull AFT model. Held out concordance is around 0.61, "
        "so treat this as a ranking signal rather than a calibrated probability."
    )


def main():
    # Header
    st.markdown('<p class="main-header">Turbofan Engine Predictive Maintenance Dashboard</p>',
                unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        result = load_models_with_metadata()

        if isinstance(result, tuple):
            models, metadata = result
            has_metadata = True
        else:
            models = result
            metadata = None
            has_metadata = False

        metrics = require_metrics()
        test_data = load_evaluation_data('test')
        raw_cycles = load_raw_cycles()

    if models is None or test_data is None:
        st.error(
            "Could not load models or data. Run `python -m src.train` from the "
            "repo root to generate them."
        )
        return

    # Show model metadata in sidebar
    if has_metadata and metadata:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### Model Information")

            for model_name, info in metadata['models'].items():
                with st.expander(display_name(model_name)):
                    st.markdown(f"**Framework:** {info.get('framework', 'unknown')}")
                    st.markdown(f"**Features:** {info.get('features', 'unknown')}")

                    for metric, value in info.get('metrics', {}).items():
                        if value is not None:
                            st.markdown(f"- {metric}: {value}")

                    st.markdown(f"**Description:** {info.get('description', '')}")
                    st.markdown(f"**Trained:** {info.get('trained_date', 'unknown')}")

    # Dataset info
    if has_metadata and 'dataset' in metadata:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### Dataset")
            st.markdown(f"**Name:** {metadata['dataset']['name']}")
            st.markdown(f"**Engines:** {metadata['dataset']['engines']}")
            st.markdown(f"**Sensors:** {metadata['dataset']['sensors']}")
            st.markdown(f"**Records:** {metadata['dataset']['total_records']}")
            if 'last_updated' in metadata:
                st.markdown(f"**Last Updated:** {metadata['last_updated']}")

    # Sidebar - Navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.selectbox(
            "Select Page",
            ["Overview", "New Prediction", "Engine Analysis", "Model Comparison",
             "Fleet Management", "Performance Metrics", "Workflow"],
            index=0
        )

        if splits := load_splits():
            st.markdown("---")
            st.caption(
                f"Evaluation pages show the {len(splits['test'])} held out test "
                f"engines. The models were fitted on {len(splits['train'])} others."
            )

    # ==================== OVERVIEW PAGE ====================
    if page == "Overview":
        st.header("System Overview")
        
        if metrics is None:
            st.stop()

        table = metrics_table(metrics, 'test')
        best_key = best_model_key(metrics, 'test')
        best = metrics['models'][best_key]
        survival_metrics = metrics.get('survival', {})

        col1, col2, col3 = st.columns(3)

        # No `delta` on any of these: Streamlit always draws a directional
        # arrow next to a delta, and none of these values have a direction.
        # The secondary figure goes in a caption instead.
        with col1:
            st.metric(label="Best model, held out test", value=display_name(best_key))
            st.caption(f"R2 {best['test']['r2']:.3f}")

        with col2:
            st.metric(label="Typical error", value=f"{best['test']['mae']:.1f} cycles")
            st.caption(f"RMSE {best['test']['rmse']:.1f}")

        with col3:
            if survival_metrics:
                ranked = max(survival_metrics.items(),
                             key=lambda kv: kv[1]['test']['concordance'])
                st.metric(label="Best risk ranking", value=display_name(ranked[0]))
                st.caption(f"C-index {ranked[1]['test']['concordance']:.3f}")
            else:
                st.metric(label="Best risk ranking", value="Not trained")
                st.caption("Run `python -m src.train` to fit the survival models")

        st.caption(
            f"Scored on {metrics['split']['test']} engines held out of training "
            f"and of model selection. Generated {metrics['generated']}."
        )

        st.markdown("---")

        st.subheader("Model Performance Comparison")

        # Formatted as strings so every row shows the same number of decimals.
        # Rounded floats render as 0.847 next to 0.8289, which reads as sloppy.
        comparison = pd.DataFrame([{
            'Model': display_name(key),
            'Validation R2': f"{m['validation']['r2']:.4f}",
            'Test R2': f"{m['test']['r2']:.4f}",
            'Test MAE': f"{m['test']['mae']:.2f}",
            'Test RMSE': f"{m['test']['rmse']:.2f}",
            '_sort': m['test']['r2'],
            'Role': MODEL_ROLES.get(key, ''),
        } for key, m in metrics['models'].items()]).sort_values(
            '_sort', ascending=False).drop(columns='_sort')

        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown(
            "Validation selected the hyperparameters and the preferred model. "
            "Test was scored once, afterwards. The gap between the two columns "
            "is the cost of having selected on validation, which is why both "
            "are shown rather than only the better one."
        )

        if survival_metrics:
            st.markdown("**Survival models** (concordance index, held out engines)")
            st.dataframe(pd.DataFrame([{
                'Model': display_name(key),
                'Validation C': f"{m['validation']['concordance']:.3f}",
                'Test C': f"{m['test']['concordance']:.3f}",
            } for key, m in survival_metrics.items()]),
                use_container_width=True, hide_index=True)
            st.caption(
                "0.5 is random ranking. With 15 engines per split the estimate "
                "is noisy, and the validation and test columns differ by more "
                "than the gap between the two models."
            )

        st.markdown("---")

        st.subheader("Which Model For Which Question")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>Priority ranking</h4>
            <p><b>Question:</b> which engines need attention first?</p>
            <p><b>Model:</b> Weibull AFT</p>
            <p><i>Ranks the fleet by risk. Does not need an accurate
            point estimate to be useful.</i></p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="warning-box">
            <h4>Scheduling</h4>
            <p><b>Question:</b> when will this engine need service?</p>
            <p><b>Model:</b> {display_name(best_key)}</p>
            <p><i>Lowest held out error, {best['test']['mae']:.1f} cycles MAE.
            Use the interval, not the point.</i></p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>Risk over a horizon</h4>
            <p><b>Question:</b> chance of failure in the next 50 cycles?</p>
            <p><b>Model:</b> Weibull AFT survival curve</p>
            <p><i>A probability over a window, which is what a
            maintenance interval actually needs.</i></p>
            </div>
            """, unsafe_allow_html=True)

    # ==================== NEW PREDICTION PAGE ====================
    elif page == "New Prediction":
        st.header("New Engine Prediction")

        # Data input options
        input_method = st.tabs(["CSV Upload", "Manual Sensor Entry"])

        with input_method[0]:
            st.subheader("Upload Sensor Data (CSV Format)")

            st.markdown("""
            **Upload a CSV file with your engine's sensor readings.**

            **Required columns:** `engine_id`, `time_cycles`, and sensor columns:
            `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`,
            `sensor_11`, `sensor_12`, `sensor_13`, `sensor_15`, `sensor_17`,
            `sensor_20`, `sensor_21`

            **Format:**
            - Multiple engines supported
            - Each row is one cycle
            - Include recent cycles for best results (at least 30 cycles for LSTM)
            """)

            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with engine sensor readings"
            )

            if uploaded_file:
                result = process_uploaded_csv(uploaded_file)

                if result:
                    df, info = result

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Engines", info['unique_engines'])
                    with col2:
                        st.metric("Total Cycles", info['total_cycles'])
                    with col3:
                        status = "Yes" if info['can_use_lstm'] else "No"
                        st.metric("Can Use LSTM", status)

                    # Select model
                    st.subheader("Select Prediction Model")
                    model_options = ['ensemble', 'gradient_boosting', 'random_forest', 'ridge']
                    if info['can_use_lstm']:
                        model_options = ['ensemble', 'lstm'] + model_options[1:]

                    model_type = st.selectbox(
                        "Choose Model",
                        model_options,
                        format_func=lambda x: {
                            'ensemble': 'Ensemble (LSTM + GB + RF) - Best',
                            'lstm': 'LSTM (Best Precision)',
                            'gradient_boosting': 'Gradient Boosting',
                            'random_forest': 'Random Forest',
                            'ridge': 'Ridge Regression'
                        }[x]
                    )

                    use_ensemble = (model_type == 'ensemble')

                    if st.button("Generate Prediction", type="primary"):
                        # Handle ensemble prediction
                        if use_ensemble and info['can_use_lstm']:
                            # Engineer the first engine's raw data into the
                            # 153-feature space, then let the ensemble helper
                            # select the LSTM subset internally.
                            first_engine = df[df['engine_id'] == df['engine_id'].iloc[0]]
                            model_cols = _model_feature_columns()
                            engine_features = engineer_for_prediction(first_engine)[model_cols]

                            ensemble_result = predict_ensemble_rul(engine_features, models, metrics)
                            if ensemble_result:
                                rul, individual_preds, ens_weights = ensemble_result
                                interval_key = best_model_key(metrics) if metrics else None
                            else:
                                rul, individual_preds, ens_weights = None, None, None
                                interval_key = None
                        else:
                            resolved = model_type if not use_ensemble else 'gradient_boosting'
                            rul, latest_data = generate_new_prediction(df, models, resolved)
                            individual_preds = None
                            ens_weights = None
                            interval_key = resolved

                        if rul is not None:
                            risk_label, risk_class = classify_risk(rul)

                            # Interval from this model's held out residuals.
                            interval = get_prediction_interval(rul, interval_key, metrics)

                            # Display results
                            st.markdown("---")
                            st.subheader("Prediction Results")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if risk_class == "critical":
                                    st.markdown(f"""
                                    <div class="critical-box">
                                    <h3>Predicted RUL</h3>
                                    <h2>{rul:.1f} cycles</h2>
                                    <p>{risk_label}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                elif risk_class == "warning":
                                    st.markdown(f"""
                                    <div class="warning-box">
                                    <h3>Predicted RUL</h3>
                                    <h2>{rul:.1f} cycles</h2>
                                    <p>{risk_label}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="success-box">
                                    <h3>Predicted RUL</h3>
                                    <h2>{rul:.1f} cycles</h2>
                                    <p>{risk_label}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with col2:
                                st.metric(
                                    "Expected Failure",
                                    f"In ~{rul:.0f} cycles"
                                )

                            with col3:
                                st.metric(
                                    "Model Used",
                                    "Ensemble - Best" if use_ensemble else model_type.replace('_', ' ').title()
                                )

                            render_prediction_interval(rul, interval, interval_key)
                            render_individual_predictions(individual_preds, ens_weights)
                            render_survival_section(df, models)

                            # SHAP explainability
                            shap_model = 'gradient_boosting' if use_ensemble else model_type
                            if shap_model in ['gradient_boosting', 'random_forest', 'ridge']:
                                st.markdown("---")
                                st.subheader("Feature Importance (SHAP)")

                                first_engine = df[df['engine_id'] == df['engine_id'].iloc[0]]
                                model_cols = _model_feature_columns()
                                engine_data = engineer_for_prediction(first_engine)[model_cols]

                                shap_results = get_shap_explanations(engine_data, models, shap_model, n_features=5)
                                
                                if shap_results:
                                    shap_df = pd.DataFrame(list(shap_results.items()), columns=['Feature', 'Importance'])
                                    shap_df = shap_df.sort_values('Importance', ascending=True)
                                    
                                    fig_shap = px.bar(
                                        shap_df,
                                        x='Importance',
                                        y='Feature',
                                        orientation='h',
                                        title='Top 5 Most Influential Features',
                                        color='Importance',
                                        color_continuous_scale='Blues'
                                    )
                                    fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_shap, use_container_width=True)
                                    
                                    st.caption("Higher importance values indicate greater influence on the prediction")

                            # Maintenance recommendation
                            st.subheader("Maintenance Recommendation")

                            if rul < 30:
                                st.error(f"""
                                **IMMEDIATE ACTION REQUIRED**

                                - Schedule maintenance within the next {int(rul * 0.7)} cycles
                                - Prepare replacement parts
                                - Monitor engine closely
                                """)
                            elif rul < 60:
                                st.warning(f"""
                                **PLAN MAINTENANCE SOON**

                                - Schedule maintenance within {int(rul * 0.8)} cycles
                                - Order replacement parts
                                - Increase monitoring frequency
                                """)
                            else:
                                st.success(f"""
                                **ENGINE HEALTHY**

                                - Next maintenance in approximately {int(rul * 0.8)} cycles
                                - Continue routine monitoring
                                - Engine performing within normal parameters
                                """)

        with input_method[1]:
            st.subheader("Manual Sensor Entry")

            st.markdown("""
            Enter the latest sensor readings for an engine.
            This uses the Gradient Boosting model for prediction.

            The entered readings are treated as a steady-state engine history
            (the engine has been reading these stable values), which lets us
            compute the rolling/lag/trend features the model expects.

            **Note:** For multi-cycle history, use CSV upload instead.
            """)

            # Typical FD001 cruise readings - used as sensible defaults so the
            # form predicts something meaningful instead of all-zero sensors.
            sensor_defaults = {
                2: 642.64, 3: 1590.10, 4: 1408.04, 7: 553.44, 8: 2388.09,
                9: 9060.66, 11: 47.51, 12: 521.48, 13: 2388.09, 15: 8.44,
                17: 393.00, 20: 38.83, 21: 23.30,
            }

            # Create form for manual entry
            with st.form("manual_sensor_form"):
                st.markdown("#### Sensor Readings")

                sensor_inputs = {}
                sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]

                cols = st.columns(4)
                for i, sensor_id in enumerate(sensors):
                    with cols[i % 4]:
                        sensor_inputs[f'sensor_{sensor_id}'] = st.number_input(
                            f"Sensor {sensor_id}",
                            value=float(sensor_defaults[sensor_id]),
                            step=0.01,
                            key=f"sensor_{sensor_id}"
                        )

                submitted = st.form_submit_button("Predict RUL", type="primary")

                if submitted:
                    # Build a steady-state history from the entered sensors and
                    # engineer it into the model's 153-feature space.
                    entered = {f'sensor_{s}': sensor_inputs[f'sensor_{s}'] for s in sensors}
                    history = fe.make_steady_state_history(entered, n_cycles=30)
                    model_cols = _model_feature_columns()
                    df = fe.prepare_model_features(history)[model_cols]

                    # Get prediction (last cycle of the steady-state history)
                    rul = models['gradient_boosting'].predict(df.iloc[-1:].values)[0]
                    rul = max(0, rul)
                    risk_label, risk_class = classify_risk(rul)

                    # Display results
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        if risk_class == "critical":
                            st.markdown(f"""
                            <div class="critical-box">
                            <h3>Predicted RUL</h3>
                            <h2>{rul:.1f} cycles</h2>
                            <p>{risk_label}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif risk_class == "warning":
                            st.markdown(f"""
                            <div class="warning-box">
                            <h3>Predicted RUL</h3>
                            <h2>{rul:.1f} cycles</h2>
                            <p>{risk_label}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box">
                            <h3>Predicted RUL</h3>
                            <h2>{rul:.1f} cycles</h2>
                            <p>{risk_label}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.metric("Expected Failure", f"In ~{rul:.0f} cycles")
                    
                    render_prediction_interval(
                        rul, get_prediction_interval(rul, 'gradient_boosting', metrics),
                        'gradient_boosting')

                    # SHAP explainability
                    st.markdown("---")
                    st.subheader("Feature Importance (SHAP)")

                    shap_results = get_shap_explanations(df, models, 'gradient_boosting', n_features=5)
                    
                    if shap_results:
                        shap_df = pd.DataFrame(list(shap_results.items()), columns=['Feature', 'Importance'])
                        shap_df = shap_df.sort_values('Importance', ascending=True)
                        
                        fig_shap = px.bar(
                            shap_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 5 Most Influential Features',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig_shap, use_container_width=True)
                        
                        st.caption("Higher importance values indicate greater influence on the prediction")

                    # Manual entry synthesises a steady-state history, which
                    # never reaches the survival model's landmark, so this
                    # correctly renders the "not enough history" message rather
                    # than a fabricated curve.
                    render_survival_section(df, models)

                    # Maintenance recommendation
                    st.subheader("Maintenance Recommendation")

                    if rul < 30:
                        st.error(f"""
                        **IMMEDIATE ACTION REQUIRED**

                        - Schedule maintenance within the next {int(rul * 0.7)} cycles
                        - Prepare replacement parts
                        - Monitor engine closely
                        """)
                    elif rul < 60:
                        st.warning(f"""
                        **PLAN MAINTENANCE SOON**

                        - Schedule maintenance within {int(rul * 0.8)} cycles
                        - Order replacement parts
                        - Increase monitoring frequency
                        """)
                    else:
                        st.success(f"""
                        **ENGINE HEALTHY**

                        - Next maintenance in approximately {int(rul * 0.8)} cycles
                        - Continue routine monitoring
                        - Engine performing within normal parameters
                        """)

    # ==================== ENGINE ANALYSIS PAGE ====================
    elif page == "Engine Analysis":
        st.header("Individual Engine Analysis")
        
        # Get unique engines
        if 'engine_id' in test_data.columns:
            engine_ids = sorted(test_data['engine_id'].unique())
            
            selected_engine = st.selectbox("Select Engine ID", engine_ids)
            
            # Filter data for selected engine
            engine_data = test_data[test_data['engine_id'] == selected_engine].copy()
            
            # Select sensor features (exclude ID, cycle, RUL columns)
            feature_cols = [col for col in engine_data.columns 
                          if col not in ['engine_id', 'time_cycles', 'RUL', 'target']]
            
            engine_features = engine_data[feature_cols]
            
            # Generate predictions from all models
            st.subheader("RUL Predictions from All Models")
            
            # Get ensemble prediction
            ensemble_result = predict_ensemble_rul(engine_features, models, metrics)
            if ensemble_result:
                ensemble_rul, individual_preds, ens_weights = ensemble_result
            else:
                ensemble_rul = predict_rul(engine_features, models, 'lstm')
                individual_preds = {'lstm': ensemble_rul}
                ens_weights = {'lstm': 1.0}

            columns = st.columns(1 + len(individual_preds))

            with columns[0]:
                ensemble_risk_label, ensemble_risk_class = classify_risk(ensemble_rul)
                box = {'critical': 'critical-box', 'warning': 'warning-box'}.get(
                    ensemble_risk_class, 'success-box')
                st.markdown(f"""
                <div class="{box}">
                <h3>Ensemble</h3>
                <h2>{ensemble_rul:.1f} cycles</h2>
                <p>{ensemble_risk_label}</p>
                </div>
                """, unsafe_allow_html=True)

            for col, (key, value) in zip(columns[1:], individual_preds.items()):
                with col:
                    st.metric(f"{display_name(key)} ({ens_weights.get(key, 0):.0%})",
                              f"{value:.1f} cycles",
                              delta=f"{value - ensemble_rul:+.1f} vs ensemble",
                              delta_color="off")

            st.markdown("---")

            render_prediction_interval(
                ensemble_rul,
                get_prediction_interval(ensemble_rul, best_model_key(metrics), metrics)
                if metrics else None,
                best_model_key(metrics) if metrics else None)

            st.markdown("---")

            # SHAP explainability
            st.subheader("Feature Importance (SHAP Analysis)")
            
            shap_results = get_shap_explanations(engine_features, models, 'gradient_boosting', n_features=5)
            
            if shap_results:
                shap_col1, shap_col2 = st.columns([2, 1])
                
                with shap_col1:
                    shap_df = pd.DataFrame(list(shap_results.items()), columns=['Feature', 'Importance'])
                    shap_df = shap_df.sort_values('Importance', ascending=True)
                    
                    fig_shap = px.bar(
                        shap_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 5 Most Influential Features for This Engine',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_shap, use_container_width=True)
                
                with shap_col2:
                    st.markdown("""
                    **What this means:**
                    
                    The SHAP (SHapley Additive exPlanations) values show which sensors are driving the current RUL prediction.
                    
                    - **Higher importance**: This sensor strongly affects the predicted RUL
                    - **Positive vs negative**: Direction of impact (not shown here)
                    - **Interpretability**: Helps engineers understand what to monitor
                    """)
            
            st.markdown("---")

            # Survival curve for this engine, from its own raw cycle history.
            engine_raw = (raw_cycles[raw_cycles['engine_id'] == selected_engine]
                          if raw_cycles is not None else None)
            render_survival_section(engine_raw, models)

            lstm_rul = individual_preds.get(
                'lstm', predict_rul(engine_features, models, 'lstm'))

            st.markdown("---")

            # Maintenance recommendations
            st.subheader("Maintenance Recommendations")

            if lstm_rul < 30:
                st.error(f"""
                **IMMEDIATE ACTION REQUIRED**
                - Schedule maintenance within the next {int(lstm_rul * 0.7)} cycles
                - Prepare replacement parts
                - Monitor engine closely
                - Consider temporary operational restrictions
                """)
            elif lstm_rul < 60:
                st.warning(f"""
                **PLAN MAINTENANCE SOON**
                - Schedule maintenance within {int(lstm_rul * 0.8)} cycles
                - Order replacement parts
                - Increase monitoring frequency
                """)
            else:
                st.success(f"""
                **ENGINE HEALTHY**
                - Next maintenance in approximately {int(lstm_rul * 0.8)} cycles
                - Continue routine monitoring
                - Engine performing within normal parameters
                """)
            
            # Sensor trends
            st.subheader("Sensor Readings Trends")

            # Raw sensor columns only (exclude operational settings and engineered features)
            sensor_options = [c for c in feature_cols if c in fe.RAW_SENSORS][:10]
            selected_sensors = st.multiselect(
                "Select sensors to visualize",
                sensor_options,
                default=sensor_options[:3]
            )
            
            if selected_sensors and 'time_cycles' in engine_data.columns:
                fig_sensors = go.Figure()
                
                for sensor in selected_sensors:
                    fig_sensors.add_trace(go.Scatter(
                        x=engine_data['time_cycles'],
                        y=engine_data[sensor],
                        mode='lines',
                        name=sensor
                    ))
                
                fig_sensors.update_layout(
                    title="Sensor Readings Over Engine Lifetime",
                    xaxis_title="Time (cycles)",
                    yaxis_title="Sensor Value (normalized)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_sensors, use_container_width=True)
        else:
            st.warning("Engine ID column not found in test data.")
    
    # ==================== MODEL COMPARISON PAGE ====================
    elif page == "Model Comparison":
        st.header("Model Performance Comparison")
        
        # Performance metrics visualization
        st.subheader("Prediction Accuracy Comparison")
        
        if metrics is None:
            st.stop()

        table = metrics_table(metrics, 'test')

        fig_comparison = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Test R2 (higher is better)",
                            "Test MAE in cycles (lower is better)")
        )
        fig_comparison.add_trace(
            go.Bar(x=table['Model'], y=table['R2'], marker_color='#1f77b4'),
            row=1, col=1)
        fig_comparison.add_trace(
            go.Bar(x=table['Model'], y=table['MAE'], marker_color='#ff7f0e'),
            row=1, col=2)
        fig_comparison.update_layout(height=420, showlegend=False)
        fig_comparison.update_yaxes(title_text="R2", row=1, col=1)
        fig_comparison.update_yaxes(title_text="MAE (cycles)", row=1, col=2)
        st.plotly_chart(fig_comparison, use_container_width=True)

        st.markdown("---")
        st.subheader("Validation Against Test")

        gap = pd.DataFrame([{
            'Model': display_name(key),
            'Validation R2': m['validation']['r2'],
            'Test R2': m['test']['r2'],
            'Gap': m['validation']['r2'] - m['test']['r2'],
        } for key, m in metrics['models'].items()]).sort_values(
            'Test R2', ascending=False)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(x=gap['Model'], y=gap['Validation R2'],
                                 name='Validation', marker_color='#aec7e8'))
        fig_gap.add_trace(go.Bar(x=gap['Model'], y=gap['Test R2'],
                                 name='Test', marker_color='#1f77b4'))
        fig_gap.update_layout(barmode='group', height=400, yaxis_title='R2')
        st.plotly_chart(fig_gap, use_container_width=True)

        st.markdown(
            "Validation picked the hyperparameters and the preferred model, so "
            "its score is optimistic by construction. Test was scored once, "
            "after every choice had been made. Reporting the validation number "
            "as if it estimated generalisation is the single most common way "
            "these projects overstate themselves."
        )

        st.markdown("---")
        st.subheader("Improvement Over the Linear Baseline")

        if 'linear_regression' in metrics['models']:
            baseline = metrics['models']['linear_regression']['test']
            best = metrics['models'][best_model_key(metrics)]['test']

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "MAE reduction vs linear regression",
                    f"{(baseline['mae'] - best['mae']) / baseline['mae'] * 100:.1f}%",
                    delta=f"{best['mae'] - baseline['mae']:.2f} cycles",
                )
            with col2:
                st.metric(
                    "R2 gain vs linear regression",
                    f"{best['r2'] - baseline['r2']:+.4f}",
                    delta=f"from {baseline['r2']:.4f} to {best['r2']:.4f}",
                    delta_color="off",
                )

            st.caption(
                "The baseline exists to make the complex models justify their "
                "cost. That gap is real but modest, which is worth knowing "
                "before reaching for a sequence model in production."
            )

        st.markdown("---")
        st.subheader("Trade-offs")

        st.dataframe(pd.DataFrame({
            'Model': table['Model'],
            'R2': table['R2'].map('{:.4f}'.format),
            'MAE': table['MAE'].map('{:.2f}'.format),
            'RMSE': table['RMSE'].map('{:.2f}'.format),
            'NASA score': table['NASA score'].map('{:,.0f}'.format),
            'Role': table['Role'],
        }), use_container_width=True, hide_index=True)

        st.caption(
            "NASA score is the CMAPSS asymmetric penalty: late predictions cost "
            "more than early ones, because an optimistic error is the one that "
            "strands an aircraft. Note it does not always rank models the same "
            "way R2 does."
        )
    
    # ==================== FLEET MANAGEMENT PAGE ====================
    elif page == "Fleet Management":
        st.header("Fleet-Wide Risk Assessment")
        
        if 'engine_id' in test_data.columns:
            engine_ids = sorted(test_data['engine_id'].unique())
            
            # Calculate RUL for all engines
            fleet_predictions = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, engine_id in enumerate(engine_ids):
                engine_data = test_data[test_data['engine_id'] == engine_id].copy()
                feature_cols = [col for col in engine_data.columns 
                              if col not in ['engine_id', 'time_cycles', 'RUL', 'target']]
                engine_features = engine_data[feature_cols]
                
                lstm_rul = predict_rul(engine_features, models, 'lstm')
                gb_rul = predict_rul(engine_features, models, 'gradient_boosting')
                risk_label, risk_class = classify_risk(lstm_rul)
                
                fleet_predictions.append({
                    'Engine ID': engine_id,
                    'LSTM RUL': lstm_rul,
                    'GB RUL': gb_rul,
                    'Risk Level': risk_label,
                    'Risk Class': risk_class,
                    'Cycles Remaining': int(lstm_rul)
                })
                
                progress_bar.progress((idx + 1) / len(engine_ids))
                status_text.text(f"Analyzing engine {idx + 1} of {len(engine_ids)}")
            
            progress_bar.empty()
            status_text.empty()
            
            fleet_df = pd.DataFrame(fleet_predictions)
            
            # Fleet summary metrics
            st.subheader("Fleet Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            critical_count = len(fleet_df[fleet_df['Risk Class'] == 'critical'])
            warning_count = len(fleet_df[fleet_df['Risk Class'] == 'warning'])
            healthy_count = len(fleet_df[fleet_df['Risk Class'] == 'success'])
            
            with col1:
                st.metric("Total Engines", len(fleet_df))
            
            with col2:
                st.metric("Critical", critical_count,
                         delta=f"{(critical_count/len(fleet_df)*100):.1f}%")

            with col3:
                st.metric("Warning", warning_count,
                         delta=f"{(warning_count/len(fleet_df)*100):.1f}%")

            with col4:
                st.metric("Healthy", healthy_count,
                         delta=f"{(healthy_count/len(fleet_df)*100):.1f}%")
            
            # Risk distribution
            st.subheader("Risk Distribution")
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Critical', 'Warning', 'Healthy'],
                values=[critical_count, warning_count, healthy_count],
                marker_colors=['#dc3545', '#ffc107', '#28a745'],
                hole=0.3
            )])
            
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Priority list
            st.subheader("Priority Maintenance Schedule")
            
            # Sort by RUL (ascending)
            priority_df = fleet_df.sort_values('LSTM RUL').head(20)

            # Color code by risk
            def color_risk(val):
                if 'CRITICAL' in str(val):
                    return 'background-color: #f8d7da'
                elif 'WARNING' in str(val):
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #d4edda'

            # Styler.map, not applymap: the latter was removed in pandas 3.
            styled_df = priority_df[
                ['Engine ID', 'LSTM RUL', 'Risk Level', 'Cycles Remaining']
            ].style.map(color_risk, subset=['Risk Level'])
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # RUL distribution
            st.subheader("RUL Distribution Across Fleet")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=fleet_df['LSTM RUL'],
                nbinsx=30,
                marker_color='#1f77b4',
                name='RUL Distribution'
            ))
            
            fig_hist.update_layout(
                xaxis_title="Remaining Useful Life (cycles)",
                yaxis_title="Number of Engines",
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Engine ID column not found in test data.")
    
    # ==================== PERFORMANCE METRICS PAGE ====================
    elif page == "Performance Metrics":
        st.header("Detailed Performance Analysis")

        if metrics is None:
            st.stop()

        table = metrics_table(metrics, 'test')

        st.subheader("Held Out Test Metrics")
        st.dataframe(pd.DataFrame({
            'Model': table['Model'],
            'R2': table['R2'].map('{:.4f}'.format),
            'MAE': table['MAE'].map('{:.2f}'.format),
            'RMSE': table['RMSE'].map('{:.2f}'.format),
            'NASA score': table['NASA score'].map('{:,.0f}'.format),
        }), use_container_width=True, hide_index=True)

        st.caption(
            f"{metrics['split']['test']} engines, {metrics['n_features']} features, "
            f"seed {metrics['seed']}. Generated {metrics['generated']}."
        )

        st.markdown("---")
        st.subheader("Error Distribution")

        st.markdown(
            "These are the residual quantiles measured on held out engines, and "
            "they are what the prediction intervals elsewhere in this dashboard "
            "are built from. A positive residual means the model predicted more "
            "remaining life than the engine had, which is the dangerous direction."
        )

        residual_rows = []
        for key, m in metrics['models'].items():
            r = m.get('residuals_test')
            if not r:
                continue
            residual_rows.append({
                'Model': display_name(key),
                'p2.5': round(r['p2.5'], 1),
                'Median': round(r['p50'], 1),
                'p97.5': round(r['p97.5'], 1),
                'Std': round(r['std'], 1),
                'Interval width': round(r['p97.5'] - r['p2.5'], 1),
            })

        if residual_rows:
            residuals = pd.DataFrame(residual_rows)
            st.dataframe(residuals, use_container_width=True, hide_index=True)

            fig_resid = go.Figure()
            for row in residual_rows:
                fig_resid.add_trace(go.Bar(
                    x=[row['Interval width']], y=[row['Model']],
                    orientation='h', name=row['Model'], showlegend=False,
                    marker_color='#1f77b4'))
            fig_resid.update_layout(
                height=340,
                xaxis_title='95% interval width (cycles)',
                title='Narrower is more useful, provided it is honest')
            st.plotly_chart(fig_resid, use_container_width=True)

            st.caption(
                "A median residual away from zero means the model is biased in "
                "that direction, which matters more than the width for planning."
            )

        st.markdown("---")
        st.subheader("Cost and Interpretability")

        st.markdown(
            "Accuracy is not the only axis, and the table above deliberately "
            "does not rank on it alone."
        )

        st.dataframe(pd.DataFrame([
            {'Model': 'LSTM', 'Training cost': 'High',
             'Input needed': '30 cycles of history',
             'Interpretability': 'Low, no direct feature attribution'},
            {'Model': 'Gradient Boosting', 'Training cost': 'Low',
             'Input needed': 'One cycle',
             'Interpretability': 'High, feature importance and SHAP'},
            {'Model': 'Random Forest', 'Training cost': 'Low',
             'Input needed': 'One cycle',
             'Interpretability': 'High, feature importance and SHAP'},
            {'Model': 'Ridge / Lasso', 'Training cost': 'Very low',
             'Input needed': 'One cycle',
             'Interpretability': 'High, signed coefficients'},
            {'Model': 'Weibull AFT / Cox', 'Training cost': 'Very low',
             'Input needed': '100 cycles to the landmark',
             'Interpretability': 'High, and gives a probability not a point'},
        ]), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("How to Read These Numbers")

        best_key = best_model_key(metrics)
        best = metrics['models'][best_key]['test']
        st.markdown(f"""
        **{display_name(best_key)} leads on held out R2 at {best['r2']:.3f}**, with
        {best['mae']:.1f} cycles of mean absolute error. The margin over gradient
        boosting is small, and gradient boosting trains in seconds, needs a single
        cycle rather than thirty, and can explain any individual prediction. That
        is a real trade rather than an obvious win.

        **The test column is the one to quote.** Validation chose the
        hyperparameters and the preferred model, so it is optimistic by
        construction. Both are shown so the size of that effect is visible.

        **{metrics['split']['test']} engines is a small sample.** The gaps between
        the top three models are comparable to the uncertainty in the estimates
        themselves. Cross validation over engine folds would be the honest next
        step, and it is listed as future work rather than claimed as done.
        """)

    elif page == "Workflow":
        st.header("How This Was Built")

        st.markdown(
            "This page documents the pipeline behind the numbers, including the "
            "parts that were wrong the first time. The ordering decisions matter "
            "more than the model choices, and most of them are only visible here."
        )

        st.subheader("Pipeline")

        st.code("""
raw CMAPSS files  (data/bronze, tracked in git, never written to)
        |
        v
  clean sensors   drop 4 constant, 3 low variance, 1 redundant
        |         -> data/silver
        v
  SPLIT BY ENGINE            <-- before anything is fitted
        |                        -> splits.json
        +-- train (70) -> FeaturePipeline.fit()
        |                  correlation filter + scaler learned here only
        |                  -> feature_pipeline.joblib
        |
        +-- val (15)   -> tune hyperparameters
        |
        +-- test (15)  -> scored once, at the end
                           -> metrics.json
        """, language=None)

        st.markdown("""
        Two things in that diagram are the whole point.

        **The split is by engine, not by row.** Rows here are cycles. Two rows
        from one engine share rolling windows, lag features and a single
        degradation trajectory, so a random row split puts near duplicates on
        both sides and measures nothing useful.

        **The split happens before anything is fitted.** Rolling windows and lags
        are computed inside one engine's history, so they are safe to build
        early. The correlation filter and the scaler are *learned from data*, so
        they see training engines only. Getting this backwards is invisible in
        the code and shows up only as a score that is quietly too good.
        """)

        st.markdown("---")
        st.subheader("Stages")

        stages = pd.DataFrame([
            {'Stage': '1. Ingestion',
             'What happens': 'Read raw files, attach RUL from the engine last cycle or the truth file',
             'Artifact': 'data/bronze'},
            {'Stage': '2. Cleaning',
             'What happens': 'Drop constant, low variance and redundant sensors',
             'Artifact': 'data/silver'},
            {'Stage': '3. Split',
             'What happens': 'Partition engines 70/15/15, written down so nothing recomputes it',
             'Artifact': 'splits.json'},
            {'Stage': '4. Features',
             'What happens': 'Rolling, lag, trend and EWMA per engine, then a train-only correlation filter and scaler',
             'Artifact': 'feature_pipeline.joblib'},
            {'Stage': '5. Baselines',
             'What happens': 'Linear, Ridge, Lasso, so complex models have something to beat',
             'Artifact': '*.pkl'},
            {'Stage': '6. Models',
             'What happens': 'Random Forest, Gradient Boosting, LSTM on the same split',
             'Artifact': '*.pkl, *.keras'},
            {'Stage': '7. Survival',
             'What happens': 'Weibull AFT and Cox on a landmark design with censoring',
             'Artifact': 'waft.pkl, cph.pkl'},
            {'Stage': '8. Evaluation',
             'What happens': 'Test scored once, with residual quantiles kept for intervals',
             'Artifact': 'metrics.json'},
        ])
        st.dataframe(stages, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("What Went Wrong the First Time")

        st.markdown("""
        Listing these is more useful than a list of achievements, because every
        one of them produced a number that looked fine.

        | Problem | Effect |
        |---|---|
        | Scaler and correlation filter fit before the split | Held out engines leaked into every feature |
        | Test set built and never scored | Reported R2 was a validation score that had also picked the model |
        | Two notebooks split engines differently | LSTM trained on engines the tree models were tested on |
        | LSTM scaler never saved | Served the network inputs on a scale it was never trained on |
        | Survival covariates taken from the failure point | Model read the answer; concordance was also in-sample |
        | Prediction intervals from `np.random.normal` | A fabricated band labelled "95% confidence" |
        | Metrics hardcoded in four places | They disagreed with each other and with the README |

        All of these are fixed, and the fixes are pinned by tests. The leakage
        one is worth spelling out: the test mutates the held out engines by a
        factor of 1000 and asserts that nothing the pipeline learned moves.
        """)

        st.markdown("---")
        st.subheader("Reproducing This")

        st.code("pip install -r requirements.txt\n"
                "python -m src.train\n"
                "streamlit run webapp/dashboard.py", language='bash')

        if metrics:
            st.caption(
                f"The current artifacts were generated {metrics['generated']} "
                f"in {metrics.get('training_seconds', 0):.0f} seconds, "
                f"seed {metrics['seed']}."
            )

        st.markdown("""
        Tree models reproduce exactly from the seed. TensorFlow on CPU does not,
        so the LSTM moves by around a point of R2 between environments. Stated
        rather than hidden.

        **On benchmarks.** An earlier version of this page compared these results
        against published CMAPSS papers and claimed the comparison used "the same
        test set". It did not: this project holds out 15 engines from the training
        file, while the literature reports on the official FD001 test split, and
        usually as RMSE and the NASA score rather than R2. Those numbers also had
        no citations attached. The chart has been removed rather than patched,
        because a comparison that cannot be sourced is worse than no comparison.
        Evaluating on the official test split is listed as future work in the
        README.
        """)

        st.markdown("---")
        st.info(
            "Self-educational portfolio project. The pipeline, the tests and "
            "the list of things that were wrong the first time are in the "
            "repository README."
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>Turbofan Engine Predictive Maintenance, CMAPSS FD001</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
