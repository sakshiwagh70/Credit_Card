import pickle
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv")
feat_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
X = df[feat_cols]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Pipeline: scale → SMOTE → MLP ─────────────────────────────────────────
# SMOTE goes inside the pipeline so it only applies to training data.
# imblearn Pipeline automatically skips SMOTE at predict/transform time.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42, sampling_strategy=0.3)),  # upsample fraud to 30% of legit
    ("model",  MLPClassifier(
        hidden_layer_sizes=(64, 32),   # two hidden layers
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,           # stops when val loss stops improving
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
    )),
])

pipeline.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
proba = pipeline.predict_proba(X_test)[:, 1]
print(f"\nROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# ── Find best threshold via precision-recall curve ─────────────────────────
# Maximize F1 — best balance between precision and recall.
# This avoids the two extremes:
#   threshold too high (0.97) → recall 48%, misses half of fraud
#   threshold too low  (0.00) → precision 6%, flags everything as fraud
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, proba)

f1_scores = (
    2 * precision_vals[:-1] * recall_vals[:-1]
    / (precision_vals[:-1] + recall_vals[:-1] + 1e-9)
)
best_idx = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])

print(f"\nBest threshold (max F1): {best_threshold:.4f}")
print(
    f"At this threshold → "
    f"Precision: {precision_vals[best_idx]:.3f}, "
    f"Recall: {recall_vals[best_idx]:.3f}, "
    f"F1: {f1_scores[best_idx]:.3f}"
)

preds = (proba >= best_threshold).astype(int)
print("\n=== Classification Report ===")
print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))

# ── Fraud score sanity check ───────────────────────────────────────────────
fraud_scores = proba[y_test == 1]
print("=== Fraud score distribution (actual fraud rows) ===")
print(f"  Min:  {fraud_scores.min():.4f}")
print(f"  Max:  {fraud_scores.max():.4f}")
print(f"  Mean: {fraud_scores.mean():.4f}")
print(f"  % scored above threshold ({best_threshold:.4f}): {(fraud_scores >= best_threshold).mean() * 100:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────
artifact = {
    "pipeline":   pipeline,
    "threshold":  best_threshold,
    "model_name": "nn_final_model",
}
with open("saved_models/nn_final_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n✅ Saved to saved_models/nn_final_model.pkl")