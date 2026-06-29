"""
Machine Learning Pipeline: Train and Test Split
================================================
Demonstrates splitting data, training a model, and evaluating performance
using accuracy, precision, recall, and F1-score.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("  ML Pipeline: Train/Test Split Demo (Iris Dataset)")
print("=" * 60)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
target_names = iris.target_names

print(f"\n[1] Dataset loaded")
print(f"    Samples  : {X.shape[0]}")
print(f"    Features : {X.shape[1]}  →  {list(X.columns)}")
print(f"    Classes  : {list(target_names)}")
print(f"\n    First 5 rows:\n{X.head()}")

# ── 2. Train / Test Split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"\n[2] Train/Test Split  (test_size=0.1, random_state=42)")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ── 3. Feature scaling ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
X_test_scaled  = scaler.transform(X_test)        # transform test with same params

print(f"\n[3] Feature scaling applied (StandardScaler)")

# ── 4. Model training ────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"\n[4] Model trained: LogisticRegression")

# ── 5. Prediction & Evaluation ───────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\n[5] Evaluation Metrics")
print(f"    {'Metric':<12} {'Score':>8}")
print(f"    {'-'*22}")
print(f"    {'Accuracy':<12} {accuracy:>8.4f}")
print(f"    {'Precision':<12} {precision:>8.4f}")
print(f"    {'Recall':<12} {recall:>8.4f}")
print(f"    {'F1-Score':<12} {f1:>8.4f}")

print(f"\n    Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# ── 6. Save metrics to CSV ───────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score":  [accuracy, precision, recall, f1],
})
os.makedirs("outputs", exist_ok=True)
metrics_df.to_csv("outputs/metrics.csv", index=False)
print("[6] Metrics saved → outputs/metrics.csv")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ML Pipeline – Train/Test Split Results", fontsize=14, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=target_names, yticklabels=target_names,
    ax=axes[0]
)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Metrics bar chart
axes[1].bar(
    metrics_df["Metric"], metrics_df["Score"],
    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    edgecolor="white"
)
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Performance Metrics")
axes[1].set_ylabel("Score")
for i, v in enumerate(metrics_df["Score"]):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("outputs/results.png", dpi=150, bbox_inches="tight")
plt.close()
print("[7] Plot saved     → outputs/results.png")

print("\n✅ Pipeline complete.\n")
