# === Smart Campus - Occupancy Prediction (Full, with saved figures) ===
# Goal: Predict whether a room is occupied (1) or empty (0) using sensor data.
# Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn (classics only).
# Includes: Overfitting checks, Train vs Test metrics, ROC-AUC, warnings for suspiciously high accuracy.
# Saves all figures into ./visualizations as PNG (300 dpi).

import os
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save-only backend to avoid GUI requirement
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

# Reproducibility
RANDOM_STATE = 42

# ---- Figure saving helpers ----
VIS_DIR = Path("visualizations")
VIS_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name: str):
    """Save current matplotlib figure into visualizations/<name>.png"""
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
    outpath = VIS_DIR / f"{safe}.png"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {outpath}")

def new_fig(figsize=(6,4)):
    plt.figure(figsize=figsize)

# -------------------
# 1) Load & Prepare Data
# -------------------
df = pd.read_csv("data.csv").dropna()

# Keep numeric columns (ignore 'date')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
assert "Occupancy" in num_cols, "Dataset must include numeric 'Occupancy' column (0/1)."

X = df[num_cols].drop(columns=["Occupancy"], errors="ignore")
y = df["Occupancy"].astype(int)

# -------------------
# 2) Train/Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# -------------------
# 3) Logistic Regression (scaled)
# -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
log_reg.fit(X_train_scaled, y_train)

y_pred_log_train = log_reg.predict(X_train_scaled)
y_pred_log_test  = log_reg.predict(X_test_scaled)
y_prob_log_test  = log_reg.predict_proba(X_test_scaled)[:, 1]

# -------------------
# 4) Random Forest (regularized)
# -------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf_train = rf.predict(X_train)
y_pred_rf_test  = rf.predict(X_test)
y_prob_rf_test  = rf.predict_proba(X_test)[:, 1]

# -------------------
# 5) Metrics & Overfitting check
# -------------------
def summarize_metrics(name, y_true_train, y_pred_train, y_true_test, y_pred_test, y_prob_test=None):
    row = {
        "model": name,
        "train_acc": accuracy_score(y_true_train, y_pred_train),
        "test_acc":  accuracy_score(y_true_test,  y_pred_test),
        "test_prec": precision_score(y_true_test, y_pred_test, zero_division=0),
        "test_rec":  recall_score(y_true_test,  y_pred_test, zero_division=0),
        "test_f1":   f1_score(y_true_test,     y_pred_test, zero_division=0),
        "test_auc":  np.nan if y_prob_test is None else roc_auc_score(y_true_test, y_prob_test),
    }
    print(f"\n=== {name} ===")
    print(f"Train Accuracy: {row['train_acc']:.3f} | Test Accuracy: {row['test_acc']:.3f}")
    print(f"Test Precision: {row['test_prec']:.3f} | Recall: {row['test_rec']:.3f} | "
          f"F1: {row['test_f1']:.3f} | AUC: {row['test_auc']:.3f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_true_test, y_pred_test, zero_division=0))
    return row

results = []
results.append(
    summarize_metrics("Logistic Regression",
                      y_train, y_pred_log_train,
                      y_test,  y_pred_log_test,
                      y_prob_test=y_prob_log_test)
)
results.append(
    summarize_metrics("Random Forest",
                      y_train, y_pred_rf_train,
                      y_test,  y_pred_rf_test,
                      y_prob_test=y_prob_rf_test)
)

summary = pd.DataFrame(results)
print("\n=== Summary (Train vs Test) ===")
print(summary.to_string(index=False))

# -------------------
# 6) Visualizations (saved to ./visualizations)
# -------------------
# Confusion Matrix (RF)
cm = confusion_matrix(y_test, y_pred_rf_test)
new_fig((5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Empty","Occupied"], yticklabels=["Empty","Occupied"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
save_fig("confusion_matrix_rf")

# ROC Curves
fig, ax = plt.subplots(figsize=(6,4))
RocCurveDisplay.from_predictions(y_test, y_prob_log_test, ax=ax)
ax.set_title("ROC Curve - Logistic Regression")
plt.tight_layout()
save_fig("roc_curve_logistic_regression")

fig, ax = plt.subplots(figsize=(6,4))
RocCurveDisplay.from_predictions(y_test, y_prob_rf_test, ax=ax)
ax.set_title("ROC Curve - Random Forest")
plt.tight_layout()
save_fig("roc_curve_random_forest")

# Feature Importance (RF)
fi = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_}) \
        .sort_values("Importance", ascending=False)
new_fig((7,4))
sns.barplot(x="Importance", y="Feature", data=fi.head(10))
plt.title("Top Feature Importances - Random Forest")
plt.tight_layout()
save_fig("feature_importances_rf_top10")

# Prediction vs Actual
new_fig((12,4))
plt.plot(y_test.values[:100], label="Actual", drawstyle="steps-post")
plt.plot(y_pred_rf_test[:100], label="Predicted", linestyle="--", drawstyle="steps-post")
plt.legend(); plt.title("Occupancy Prediction vs Actual (First 100)")
plt.xlabel("Sample"); plt.ylabel("Occupancy")
plt.tight_layout()
save_fig("pred_vs_actual_rf_first100")

# -------------------
# 7) (Optional) Learning Curves
# -------------------
PLOT_LEARNING_CURVES = True
if PLOT_LEARNING_CURVES:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def slugify(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()

    def plot_learning_curve(est, X_in, y_in, title):
        train_sizes, train_scores, val_scores = learning_curve(
            est, X_in, y_in, cv=cv, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
        )
        train_mean = train_scores.mean(axis=1)
        val_mean   = val_scores.mean(axis=1)
        new_fig((6,4))
        plt.plot(train_sizes, train_mean, marker="o", label="Train")
        plt.plot(train_sizes, val_mean,   marker="s", label="Validation")
        plt.title(f"Learning Curve — {title}")
        plt.xlabel("Training samples"); plt.ylabel("Accuracy")
        plt.legend(); plt.grid(True, alpha=0.25); plt.tight_layout()
        save_fig(f"learning_curve_{slugify(title)}")

    # Logistic Regression (scaled data)
    plot_learning_curve(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        X_train_scaled, y_train, "Logistic Regression"
    )

    # Random Forest (raw data)
    plot_learning_curve(
        RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            class_weight="balanced", max_depth=12, min_samples_leaf=3, n_jobs=-1
        ),
        X_train, y_train, "Random Forest"
    )

# -------------------
# 8) Warning for suspiciously high accuracy
# -------------------
for _, row in summary.iterrows():
    if row["test_acc"] > 0.98:
        print(f"\n⚠️ Warning: {row['model']} achieved extremely high Test Accuracy "
              f"({row['test_acc']:.3f}).")
        print("This might indicate one of the following:")
        print("- Overfitting (model memorized the training data)")
        print("- Data leakage (feature reveals Occupancy directly)")
        print("- Dataset too easy (e.g., Light highly correlated with Occupancy)")
        print("👉 Always check Precision/Recall/F1 and Train vs Test performance.")
