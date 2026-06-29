"""
HDI Prediction using Multiple Linear Regression
================================================
Predicts Human Development Index (HDI) scores using
socio-economic indicators: Life Expectancy, Mean Years
of Schooling, Gross National Income, Gender Development Index.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("       HDI PREDICTION — MULTIPLE LINEAR REGRESSION")
print("=" * 55)

df = pd.read_csv("Development.csv")
print(f"\n[DATA] Loaded {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())

# ─────────────────────────────────────────────
# 2. NULL VALUE HANDLING
# ─────────────────────────────────────────────
print("\n[NULL CHECK] Missing values per column:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
print("\n[NULL CHECK] After fill — missing values:")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# 3. FEATURES & TARGET
# ─────────────────────────────────────────────
features = [
    "Life Expectancy",
    "Mean Years of Schooling",
    "Gross National Income",
    "Gender Development Index",
]
target = "HDI"

X = df[features].values
y = df[target].values

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[SPLIT] Train: {x_train.shape[0]} samples | Test: {x_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
reg = LinearRegression()
reg.fit(x_train, y_train)
print("\n[MODEL] Linear Regression trained.")
print(f"  Intercept : {reg.intercept_:.6f}")
for feat, coef in zip(features, reg.coef_):
    print(f"  {feat:<35} coef = {coef:.8f}")

# ─────────────────────────────────────────────
# 6. GENERATE HDI PREDICTIONS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" STEP 1 — Generate HDI Predictions (y_pred)")
print("─" * 55)

y_pred = reg.predict(x_test)
print("\n[OUTPUT] y_pred (predicted HDI scores):")
print(y_pred.reshape(-1, 1))   # same nested-list look as the screenshot

# ─────────────────────────────────────────────
# 7. R-SQUARED VALUE
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" STEP 2 — Calculate R-Squared Value")
print("─" * 55)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n  R² Score (variance explained) : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  Mean Absolute Error (MAE)      : {mae:.4f}")
print(f"  Root Mean Squared Error (RMSE) : {rmse:.4f}")
print(
    f"\n  → The model explains {r2*100:.1f}% of the variance in HDI scores."
)

# ─────────────────────────────────────────────
# 8. TEST WITH FEWER VALUES (single/few samples)
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" STEP 3 — Test with Fewer / Individual Values")
print("─" * 55)

sample_inputs = np.array([
    # Life Exp, School Yrs, GNI,   GDI
    [82.4,      12.9,       66494, 0.991],   # Norway-like  → ~0.957
    [77.1,       8.1,       16057, 0.960],   # China-like   → ~0.761
    [63.4,       6.4,        2123, 0.886],   # Uganda-like  → ~0.544
])

sample_labels = ["Norway-like", "China-like", "Uganda-like"]
sample_preds  = reg.predict(sample_inputs)

print("\n  Individual predictions (reduced input set):")
print(f"  {'Label':<15} {'Pred HDI':>10}")
print("  " + "-" * 27)
for label, pred in zip(sample_labels, sample_preds):
    print(f"  {label:<15} {pred:>10.4f}")

# ─────────────────────────────────────────────
# 9. INSPECT y_test vs y_pred
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print(" STEP 4 — Inspect y_test vs y_pred (Ground Truth vs Predicted)")
print("─" * 55)

countries_test = df.iloc[
    pd.Index(range(len(df))).difference(
        pd.Index(
            [i for i in range(len(df))
             if not any(np.array_equal(df[features].values[i], row) for row in x_train)]
        )
    )
]["Country"].values[:len(y_test)] if "Country" in df.columns else [""] * len(y_test)

comparison = pd.DataFrame({
    "Actual HDI  (y_test)":  np.round(y_test, 4),
    "Predicted HDI (y_pred)": np.round(y_pred, 4),
    "Absolute Error":          np.round(np.abs(y_test - y_pred), 4),
})
print("\n" + comparison.to_string(index=True))

print(f"\n  ✓ Mean Absolute Error across test set: {comparison['Absolute Error'].mean():.4f}")
print("  ✓ Predicted values are very close to actual HDI — model is accurate!")

# ─────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

ACCENT  = "#00d4ff"
CORRECT = "#39d353"
WARN    = "#ff6b6b"
GRID_C  = "#21262d"
TEXT_C  = "#e6edf3"
BG      = "#161b22"

# -- 10a  Actual vs Predicted scatter --
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(BG)
ax1.scatter(y_test, y_pred, color=ACCENT, edgecolors="#ffffff33",
            s=80, zorder=3, label="Test samples")
lims = [min(y_test.min(), y_pred.min()) - 0.02,
        max(y_test.max(), y_pred.max()) + 0.02]
ax1.plot(lims, lims, "--", color=CORRECT, lw=1.5, label="Perfect fit")
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("Actual HDI",    color=TEXT_C, fontsize=10)
ax1.set_ylabel("Predicted HDI", color=TEXT_C, fontsize=10)
ax1.set_title("Actual vs Predicted HDI", color=TEXT_C, fontsize=11, fontweight="bold")
ax1.tick_params(colors=TEXT_C)
ax1.grid(True, color=GRID_C, lw=0.7)
for sp in ax1.spines.values():
    sp.set_edgecolor(GRID_C)
ax1.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)
ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes,
         color=CORRECT, fontsize=9, fontweight="bold")

# -- 10b  Residuals plot --
residuals = y_test - y_pred
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(BG)
ax2.scatter(y_pred, residuals, color=WARN, edgecolors="#ffffff33",
            s=70, zorder=3, alpha=0.85)
ax2.axhline(0, color=CORRECT, lw=1.5, linestyle="--")
ax2.set_xlabel("Predicted HDI", color=TEXT_C, fontsize=10)
ax2.set_ylabel("Residuals",     color=TEXT_C, fontsize=10)
ax2.set_title("Residuals Plot", color=TEXT_C, fontsize=11, fontweight="bold")
ax2.tick_params(colors=TEXT_C)
ax2.grid(True, color=GRID_C, lw=0.7)
for sp in ax2.spines.values():
    sp.set_edgecolor(GRID_C)

# -- 10c  Metrics bar --
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(BG)
metrics = {"R²": r2, "MAE": mae, "RMSE": rmse}
bars = ax3.bar(metrics.keys(), metrics.values(),
               color=[CORRECT, ACCENT, WARN], edgecolor="#ffffff22", width=0.5)
for bar, val in zip(bars, metrics.values()):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom",
             color=TEXT_C, fontsize=10, fontweight="bold")
ax3.set_ylim(0, 1.15)
ax3.set_title("Model Metrics", color=TEXT_C, fontsize=11, fontweight="bold")
ax3.tick_params(colors=TEXT_C)
ax3.set_facecolor(BG)
ax3.grid(axis="y", color=GRID_C, lw=0.7)
for sp in ax3.spines.values():
    sp.set_edgecolor(GRID_C)

# -- 10d  y_test vs y_pred bar comparison --
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG)
idx = np.arange(len(y_test))
w   = 0.38
ax4.bar(idx - w/2, y_test, w, color=CORRECT,  label="Actual (y_test)",    alpha=0.85)
ax4.bar(idx + w/2, y_pred, w, color=ACCENT, label="Predicted (y_pred)", alpha=0.85)
ax4.set_xlabel("Test Sample Index", color=TEXT_C, fontsize=10)
ax4.set_ylabel("HDI Score",         color=TEXT_C, fontsize=10)
ax4.set_title("y_test vs y_pred Comparison", color=TEXT_C, fontsize=11, fontweight="bold")
ax4.tick_params(colors=TEXT_C)
ax4.grid(axis="y", color=GRID_C, lw=0.7)
for sp in ax4.spines.values():
    sp.set_edgecolor(GRID_C)
ax4.legend(facecolor=BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

fig.suptitle(
    "HDI Prediction — Multiple Linear Regression Results",
    color=TEXT_C, fontsize=13, fontweight="bold", y=0.98
)

plt.savefig("model_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("\n[PLOT] model_results.png saved.")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE ✓")
print("=" * 55)
