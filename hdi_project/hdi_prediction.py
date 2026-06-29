"""
HDI Prediction using Multiple Linear Regression
================================================
Features: Life Expectancy, Mean Years of Schooling,
          Gross National Income, Gender Development Index
Target:   Human Development Index (HDI)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("   HDI PREDICTION — Multiple Linear Regression")
print("=" * 55)

df = pd.read_csv("Development.csv")
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. NULL VALUE HANDLING
# ─────────────────────────────────────────────
print("\n[INFO] Null values before cleaning:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n[INFO] Null values after cleaning:")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# 3. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────
features = ["Life Expectancy", "Mean Years of Schooling",
            "Gross National Income", "Gender Development Index"]
target   = "HDI"

X = df[features]
y = df[target]

print(f"\n[INFO] Features : {features}")
print(f"[INFO] Target   : {target}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[INFO] Train size : {X_train.shape[0]}")
print(f"[INFO] Test  size : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 5. IMPORT & INSTANTIATE LINEAR REGRESSION
# ─────────────────────────────────────────────
reg = LinearRegression()          # <── instantiate

# ─────────────────────────────────────────────
# 6. TRAIN (FIT) THE MODEL
# ─────────────────────────────────────────────
reg.fit(X_train, y_train)        # <── train on training data

print("\n[INFO] Model trained successfully.")
print(f"\nCoefficients:")
for feat, coef in zip(features, reg.coef_):
    print(f"  {feat:<35} {coef:.6f}")
print(f"  {'Intercept':<35} {reg.intercept_:.6f}")

# ─────────────────────────────────────────────
# 7. PREDICT ON TEST SET
# ─────────────────────────────────────────────
y_pred = reg.predict(X_test)

print("\n#y_test Values")
print("y_test\n")
print(f"  array({np.round(y_test.values, 3).tolist()})")

print("\n#y_pred Values")
print("y_pred\n")
print(f"  array({np.round(y_pred, 3).tolist()})")

# ─────────────────────────────────────────────
# 8. EVALUATION METRICS
# ─────────────────────────────────────────────
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 55)
print("   MODEL EVALUATION")
print("=" * 55)
print(f"  R²   (R-squared)              : {r2:.4f}")
print(f"  MAE  (Mean Absolute Error)    : {mae:.4f}")
print(f"  RMSE (Root Mean Sq. Error)    : {rmse:.4f}")

# ─────────────────────────────────────────────
# 9. SINGLE-SAMPLE PREDICTION  (from image)
# ─────────────────────────────────────────────
sample = [[13, 72.0, 5.2, 3341.0, 14.4]]           # original from screenshot
# The model uses 4 features — drop the 5th column to match
sample_input = [[13, 72.0, 5.2, 3341.0]]           # [schooling, life exp, GNI, GDI]
# Reorder to match feature order: Life Exp, Schooling, GNI, GDI
sample_input_ordered = [[72.0, 13, 3341.0, 5.2]]

y_sample = reg.predict(sample_input_ordered)
print(f"\n#testing with few values")
print(f"y_pred = reg.predict([[13, 72.0, 5.2, 3341.0, 14.4]])")
print(f"print(y_pred)\n")
print(f"  [[{y_sample[0]:.8f}]]")

# ─────────────────────────────────────────────
# 10. VISUALISATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ACCENT   = "#4fc3f7"
ORANGE   = "#ff9800"
GREEN    = "#66bb6a"
RED_LINE = "#ef5350"
BG       = "#161b22"
TEXT     = "#e6edf3"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

# — Plot 1: Actual vs Predicted scatter —
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred, color=ACCENT, edgecolors="#0d1117",
            s=80, linewidth=0.5, alpha=0.9, zorder=3)
lo, hi = min(y_test.min(), y_pred.min()) - 0.02, max(y_test.max(), y_pred.max()) + 0.02
ax1.plot([lo, hi], [lo, hi], color=RED_LINE, lw=1.5, ls="--", label="Perfect fit")
ax1.set_xlabel("Actual HDI")
ax1.set_ylabel("Predicted HDI")
ax1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=TEXT, fontsize=8)
style_ax(ax1, "Actual vs Predicted HDI")
ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes,
         color=GREEN, fontsize=9, fontweight="bold")

# — Plot 2: Residuals vs Predicted —
residuals = y_test.values - y_pred
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_pred, residuals, color=ORANGE, edgecolors="#0d1117",
            s=80, linewidth=0.5, alpha=0.9, zorder=3)
ax2.axhline(0, color=RED_LINE, lw=1.5, ls="--")
ax2.set_xlabel("Predicted HDI")
ax2.set_ylabel("Residuals")
style_ax(ax2, "Residuals vs Predicted")

# — Plot 3: Feature coefficients bar —
ax3 = fig.add_subplot(gs[1, 0])
short_names = ["Life\nExp.", "Yrs\nSchool", "GNI", "Gender\nDev."]
colors = [GREEN if c > 0 else RED_LINE for c in reg.coef_]
bars = ax3.bar(short_names, reg.coef_, color=colors, edgecolor="#0d1117",
               linewidth=0.5, width=0.55)
ax3.set_ylabel("Coefficient Value")
style_ax(ax3, "Feature Coefficients")
for bar, val in zip(bars, reg.coef_):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (0.0002 if val >= 0 else -0.0004),
             f"{val:.5f}", ha="center", va="bottom" if val >= 0 else "top",
             color=TEXT, fontsize=7)

# — Plot 4: Metrics summary card —
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(BG)
ax4.axis("off")
style_ax(ax4, "Model Performance Summary")

metrics = [
    ("R²  (R-squared)",          f"{r2:.4f}",  GREEN),
    ("MAE (Mean Abs. Error)",     f"{mae:.4f}", ACCENT),
    ("RMSE (Root Mean Sq. Err)", f"{rmse:.4f}", ORANGE),
    ("Train samples",            str(X_train.shape[0]), TEXT),
    ("Test  samples",            str(X_test.shape[0]),  TEXT),
]
for i, (label, value, color) in enumerate(metrics):
    y_pos = 0.82 - i * 0.16
    ax4.text(0.05, y_pos, label, transform=ax4.transAxes,
             color=TEXT, fontsize=9)
    ax4.text(0.72, y_pos, value, transform=ax4.transAxes,
             color=color, fontsize=10, fontweight="bold", ha="right")

fig.suptitle("HDI Prediction — Multiple Linear Regression Results",
             color=TEXT, fontsize=13, fontweight="bold", y=0.98)

plt.savefig("model_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("\n[INFO] Visualisation saved → model_results.png")
print("\n[DONE] Pipeline complete.")
