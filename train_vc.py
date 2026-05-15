import pickle
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# ── Define estimators ──────────────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handles imbalance natively
    random_state=42,
    n_jobs=-1,
)

nn = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,             # suppress per-iteration output for cleaner logs
)

knn = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1,
)

vc = VotingClassifier(
    estimators=[("rf", rf), ("nn", nn), ("knn", knn)],
    voting="soft",             # average predicted probabilities
)

# ── Pipeline: scale → SMOTE → VotingClassifier ────────────────────────────
# SMOTE only applied during fit; skipped automatically at predict time
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote",  SMOTE(random_state=42, sampling_strategy=0.3)),
    ("model",  vc),
])

print("Training VotingClassifier (RF + MLP + KNN) with SMOTE...")
pipeline.fit(X_train, y_train)
print("Training complete.\n")

# ── Evaluate ───────────────────────────────────────────────────────────────
proba = pipeline.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

# ── Find best threshold via F1 maximization ────────────────────────────────
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
    "model_name": "VC_final",
}
with open("saved_models/vc_final_model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n✅ Saved to saved_models/vc_final_model.pkl")