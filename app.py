import glob
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Constants
# -----------------------------
BASE_FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # 29 features
ALL_FEATURE_COLS = ["Time"] + BASE_FEATURE_COLS                    # 30 features
DEFAULT_MODEL_FILE = "vc_final_model.pkl"
MODELS_DIR = "saved_models"
DATA_PATH = os.path.join("data", "creditcard.csv")


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class LoadedArtifact:
    model: Any
    preprocessor: Any = None
    threshold: float = 0.5
    name: str = "model"
    expected_features: Optional[List[str]] = None


# -----------------------------
# Utilities
# -----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_resource(show_spinner=False)
def list_available_models() -> List[str]:
    files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl")))
    return [os.path.basename(f) for f in files]


@st.cache_resource(show_spinner=False)
def load_artifact(model_filename: str) -> LoadedArtifact:
    path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    with open(path, "rb") as f:
        artifact = pickle.load(f)

    # Raw estimator / pipeline
    if not isinstance(artifact, dict):
        loaded = LoadedArtifact(model=artifact, name=model_filename)
        loaded.expected_features = infer_expected_features(loaded)
        return loaded

    # Dict-based artifact
    model = (
        artifact.get("model")
        or artifact.get("estimator")
        or artifact.get("classifier")
        or artifact.get("pipeline")
    )
    preprocessor = (
        artifact.get("preprocessor")
        or artifact.get("scaler")
        or artifact.get("transformer")
    )
    threshold = artifact.get("threshold", 0.5)

    if model is None:
        raise ValueError(
            f"{model_filename} is a dict, but no model key was found. "
            "Expected one of: model, estimator, classifier, pipeline"
        )

    loaded = LoadedArtifact(
        model=model,
        preprocessor=preprocessor,
        threshold=float(threshold),
        name=model_filename,
    )
    loaded.expected_features = infer_expected_features(loaded)
    return loaded


def _get_n_features(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    if hasattr(obj, "n_features_in_"):
        try:
            return int(obj.n_features_in_)
        except Exception:
            pass
    return None


def _get_feature_names(obj: Any) -> Optional[List[str]]:
    if obj is None:
        return None
    if hasattr(obj, "feature_names_in_"):
        try:
            return [str(x) for x in list(obj.feature_names_in_)]
        except Exception:
            pass
    return None


def infer_expected_features(artifact: LoadedArtifact) -> List[str]:
    """Infer which raw input columns the saved object expects."""
    for obj in (artifact.preprocessor, artifact.model):
        names = _get_feature_names(obj)
        if names:
            if len(names) == 29 and "Time" not in names:
                return BASE_FEATURE_COLS.copy()
            if len(names) == 30 and "Time" in names:
                return ALL_FEATURE_COLS.copy()
            return names

    for obj in (artifact.preprocessor, artifact.model):
        n = _get_n_features(obj)
        if n == 29:
            return BASE_FEATURE_COLS.copy()
        if n == 30:
            return ALL_FEATURE_COLS.copy()

    return ALL_FEATURE_COLS.copy()


def expected_input_label(expected_features: Sequence[str]) -> str:
    if "Time" in expected_features:
        return "This model expects 30 inputs: Time, V1-V28, and Amount."
    return "This model expects 29 inputs: V1-V28 and Amount. Time is not used by this model."


def align_input_columns(df: pd.DataFrame, expected_features: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[list(expected_features)].copy()


def pretty_result_label(pred: int) -> str:
    return "Fraudulent Transaction" if int(pred) == 1 else "Legitimate Transaction"


def predict_with_artifact(
    artifact: LoadedArtifact, X: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, fraud_scores)."""
    data = X.copy()
    model = artifact.model

    if artifact.preprocessor is not None:
        data = artifact.preprocessor.transform(data)

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(data))
        fraud_prob = (
            proba[:, 1]
            if proba.ndim == 2 and proba.shape[1] >= 2
            else proba.ravel()
        )
        pred = (fraud_prob >= artifact.threshold).astype(int)
        return pred, fraud_prob

    if hasattr(model, "decision_function"):
        raw_score = np.asarray(model.decision_function(data)).ravel()
        fraud_prob = sigmoid(raw_score)
        pred = (fraud_prob >= artifact.threshold).astype(int)
        return pred, fraud_prob

    pred = np.asarray(model.predict(data)).ravel().astype(int)
    fraud_prob = np.full(len(pred), np.nan)
    return pred, fraud_prob


# -----------------------------
# Dataset loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset() -> Optional[pd.DataFrame]:
    if not os.path.exists(DATA_PATH):
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None


# -----------------------------
# Cross-model consensus demo
# -----------------------------
def _score_rows_all_models(
    df_subset: pd.DataFrame,
    expected_features: Sequence[str],
) -> Optional[np.ndarray]:
    all_model_files = list_available_models()
    score_matrix: List[np.ndarray] = []
    feat_cols = list(expected_features)

    for mf in all_model_files:
        try:
            a = load_artifact(mf)
        except Exception:
            continue

        a_feats = a.expected_features or ALL_FEATURE_COLS
        if set(a_feats) != set(feat_cols):
            continue

        try:
            X = df_subset[a_feats].copy()
            _, scores = predict_with_artifact(a, X)
            if not np.all(np.isnan(scores)):
                score_matrix.append(scores)
        except Exception:
            continue

    if not score_matrix:
        return None

    return np.nanmean(np.vstack(score_matrix), axis=0)


def get_consensus_demo_row(
    expected_features: Sequence[str],
    target: str,
    max_rows: int = 5000,
) -> Optional[Dict[str, float]]:
    df = load_dataset()
    if df is None or "Class" not in df.columns:
        return None

    feat_cols = list(expected_features)
    if not all(c in df.columns for c in feat_cols):
        return None

    target_label = 1 if target == "fraud" else 0
    subset = df[df["Class"] == target_label].copy()

    if subset.empty:
        return None

    if len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=42)

    avg_scores = _score_rows_all_models(subset, feat_cols)

    if avg_scores is None:
        return None

    best_idx = int(np.nanargmax(avg_scores)) if target_label == 1 else int(np.nanargmin(avg_scores))
    return subset.iloc[best_idx][feat_cols].to_dict()


def get_single_model_demo_row(
    artifact: LoadedArtifact,
    expected_features: Sequence[str],
    target: str,
    max_rows: int = 5000,
) -> Optional[Dict[str, float]]:
    df = load_dataset()
    if df is None or "Class" not in df.columns:
        return None

    feat_cols = list(expected_features)
    if not all(c in df.columns for c in feat_cols):
        return None

    target_label = 1 if target == "fraud" else 0
    subset = df[df["Class"] == target_label].copy()

    if subset.empty:
        return None

    if len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=42)

    X = subset[feat_cols].copy()
    try:
        preds, scores = predict_with_artifact(artifact, X)
    except Exception:
        return None

    if target_label == 1:
        correct = np.where(preds == 1)[0]
        pool = correct if len(correct) > 0 else np.arange(len(scores))
        idx = pool[int(np.nanargmax(scores[pool]))] if not np.all(np.isnan(scores)) else int(pool[0])
    else:
        correct = np.where(preds == 0)[0]
        pool = correct if len(correct) > 0 else np.arange(len(scores))
        idx = pool[int(np.nanargmin(scores[pool]))] if not np.all(np.isnan(scores)) else int(pool[0])

    return subset.iloc[int(idx)][feat_cols].to_dict()


def search_demo_sample(
    artifact: LoadedArtifact,
    expected_features: Sequence[str],
    target: str,
) -> Dict[str, float]:
    # 1. Cross-model consensus
    row = get_consensus_demo_row(expected_features, target=target)
    if row is not None:
        return row

    # 2. Single-model fallback
    row = get_single_model_demo_row(artifact, expected_features, target=target)
    if row is not None:
        return row

    # 3. Synthetic search fallback
    FRAUD_MEANS = {
        "V1": -3.0, "V2": 2.5,  "V3": -3.5, "V4":  3.5,
        "V5": -1.5, "V6": -1.5, "V7": -4.0, "V8":  0.8,
        "V9": -2.0, "V10": -4.0,"V11": 2.5, "V12": -5.0,
        "V14": -7.0,"V16": -2.5,"V17": -8.0,"V18": -2.5,
    }

    rng = np.random.default_rng(42 if target == "fraud" else 7)
    target_label = 1 if target == "fraud" else 0
    best_row: Optional[Dict[str, float]] = None
    best_score: Optional[float] = None
    candidate: Dict[str, float] = {}

    for _ in range(500):
        candidate = {}
        for col in expected_features:
            if col == "Time":
                candidate[col] = float(rng.uniform(0, 172792))
            elif col == "Amount":
                if target == "fraud":
                    candidate[col] = float(round(rng.uniform(1.0, 200.0), 2))
                else:
                    candidate[col] = float(round(max(1.0, rng.lognormal(mean=4.0, sigma=1.0)), 2))
            elif target == "fraud" and col in FRAUD_MEANS:
                candidate[col] = float(rng.normal(loc=FRAUD_MEANS[col], scale=0.8))
            else:
                candidate[col] = float(rng.normal(loc=0.0, scale=1.0))

        sample = pd.DataFrame([candidate], columns=list(expected_features))
        try:
            pred, score = predict_with_artifact(artifact, sample)
            p = float(score[0]) if not np.isnan(score[0]) else float(pred[0])
        except Exception:
            continue

        if target == "fraud":
            if best_score is None or p > best_score:
                best_score = p
                best_row = candidate
            if int(pred[0]) == target_label:
                return candidate
        else:
            if best_score is None or p < best_score:
                best_score = p
                best_row = candidate
            if int(pred[0]) == target_label:
                return candidate

    return best_row if best_row is not None else candidate


def apply_demo_sample(sample: Dict[str, float], expected_features: Sequence[str]) -> None:
    for feature in expected_features:
        st.session_state[f"single_{feature}"] = float(sample.get(feature, 0.0))


# -----------------------------
# UI
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.caption("Streamlit app for the credit card fraud ML project")

available_models = list_available_models()
if not available_models:
    st.error("No .pkl models found in the saved_models folder. Put your trained model files there first.")
    st.stop()

with st.sidebar:
    st.header("Model Settings")
    default_index = (
        available_models.index(DEFAULT_MODEL_FILE)
        if DEFAULT_MODEL_FILE in available_models
        else 0
    )
    chosen_model = st.selectbox("Choose a saved model", available_models, index=default_index)
    st.caption("The app will auto-detect whether this model expects 29 or 30 raw input features.")

    artifact = load_artifact(chosen_model)
    expected_features = artifact.expected_features or ALL_FEATURE_COLS

    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(artifact.threshold if 0.05 <= artifact.threshold <= 0.95 else 0.5),
        step=0.01,
        help="Lower threshold → more fraud alerts. Higher threshold → fewer alerts.",
    )
    artifact.threshold = threshold

    st.divider()
    st.markdown("**Detected input shape**")
    st.info(expected_input_label(expected_features))

left, right = st.columns([1.2, 0.8])

with left:
    tab_single, tab_batch, tab_about = st.tabs(["Single transaction", "Batch CSV", "About"])

    with tab_single:
        st.subheader("Predict one transaction")
        st.write("Use a demo preset or enter the features expected by the selected model.")

        demo_col1, demo_col2 = st.columns([1.3, 0.7])
        with demo_col1:
            demo_choice = st.selectbox(
                "Demo preset",
                ["Custom entry", "Likely legitimate", "Likely fraud"],
                index=0,
                help=(
                    "Picks the real dataset row that ALL models most agree on — "
                    "so every model should predict the same class."
                ),
            )
        with demo_col2:
            st.write("")
            st.write("")
            load_clicked = st.button("Load demo preset")

        if load_clicked and demo_choice != "Custom entry":
            target = "legit" if demo_choice == "Likely legitimate" else "fraud"
            with st.spinner(f"Finding best consensus {target} row across all models…"):
                demo_row = search_demo_sample(artifact, expected_features, target=target)
            apply_demo_sample(demo_row, expected_features)
            st.success(
                f"Loaded '{demo_choice}' — this row was chosen because ALL models "
                "score it most confidently as this class."
            )
            st.rerun()

        with st.form("single_predict_form"):
            input_values: Dict[str, float] = {}
            cols = st.columns(4)

            for i, feature in enumerate(expected_features):
                current_col = cols[i % 4]
                key = f"single_{feature}"
                default_value = float(st.session_state.get(key, 0.0))
                with current_col:
                    if feature == "Time":
                        input_values[feature] = st.number_input(
                            "Time",
                            value=default_value,
                            min_value=0.0,
                            step=1.0,
                            format="%.4f",
                            key=key,
                        )
                    elif feature == "Amount":
                        input_values[feature] = st.number_input(
                            "Amount",
                            value=default_value,
                            min_value=0.0,
                            step=1.0,
                            format="%.4f",
                            key=key,
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature,
                            value=default_value,
                            step=0.1,
                            format="%.6f",
                            key=key,
                        )

            submitted = st.form_submit_button("Predict")

        if submitted:
            sample = pd.DataFrame([input_values], columns=list(expected_features))
            try:
                sample = align_input_columns(sample, expected_features)
                pred, fraud_prob = predict_with_artifact(artifact, sample)
                label = pretty_result_label(int(pred[0]))

                if int(pred[0]) == 1:
                    st.error(f"Prediction: {label}")
                else:
                    st.success(f"Prediction: {label}")

                if not np.isnan(fraud_prob[0]):
                    st.metric("Fraud risk score", f"{fraud_prob[0] * 100:.2f}%")
                    st.progress(float(np.clip(fraud_prob[0], 0.0, 1.0)))

                st.dataframe(sample, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_batch:
        st.subheader("Predict from CSV")
        st.write("Upload a CSV with the same columns expected by the selected model.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.write("Preview of uploaded data")
                st.dataframe(df.head(), use_container_width=True)

                input_df = df.copy()
                if "Class" in input_df.columns:
                    input_df = input_df.drop(columns=["Class"])

                input_df = align_input_columns(input_df, expected_features)
                preds, probs = predict_with_artifact(artifact, input_df)

                out = df.copy()
                out["Predicted_Class"] = preds
                out["Prediction_Label"] = [pretty_result_label(p) for p in preds]
                if not np.all(np.isnan(probs)):
                    out["Fraud_Probability"] = probs

                st.success(f"Done. Processed {len(out)} rows.")
                st.dataframe(out.head(50), use_container_width=True)

                csv_data = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"CSV prediction failed: {e}")
                st.info("Your CSV must contain these columns: " + ", ".join(expected_features))

    with tab_about:
        st.subheader("How this app works")
        st.markdown(
            f"""
            - Load one of the saved model files from `saved_models/`
            - Enter a single transaction manually or upload a CSV
            - Predict fraud probability and final class

            **Demo preset logic**
            The "Likely fraud" and "Likely legitimate" presets score every ground-truth
            labeled row in the real dataset against **all** available models simultaneously,
            then pick the row with the highest (or lowest) **average fraud score** across
            all models. This means every model should agree on the prediction.

            **Dataset columns used by the selected model**
            {", ".join(expected_features)}

            **Important note**
            This app auto-detects whether the selected model uses 29 or 30 raw features.
            """
        )

with right:
    st.subheader("Model Info")

    st.markdown("**Selected model**")
    st.info(f"🤖  `{artifact.name}`")

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Decision threshold", f"{artifact.threshold:.2f}")
    with m2:
        st.metric("Input features", len(expected_features))

    st.divider()

    st.markdown("**Features used by this model**")
    st.caption(", ".join(expected_features))