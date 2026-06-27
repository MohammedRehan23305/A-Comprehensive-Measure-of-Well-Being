# ============================================================
#  HDI Score Predictor — Multiple Linear Regression
#  Human Development Index prediction using socio-economic data
# ============================================================

# Step 1 — Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 2 — Load dataset
# ============================================================
Development = pd.read_csv('Development.csv')
print("=" * 55)
print("  HDI Score Predictor — Multiple Linear Regression")
print("=" * 55)
print(f"\n[1] Dataset loaded: {Development.shape[0]} rows, {Development.shape[1]} columns")

# ============================================================
# Step 3 — Select features (X) and target (y)
# ============================================================

# Independent Variables (X): Inputs given to the model
# Index 2  → Country
# Index 5  → Life Expectancy
# Index 6  → Mean Years of Schooling
# Index 7  → Gross National Income (GNI) per capita
# Index 67 → Internet Users  (or Gender Development Index)
X = Development.iloc[:, [2, 5, 6, 7, 67]]
X = pd.DataFrame(X)

# Dependent Variable (y): HDI Score (column index 4)
y = Development.iloc[:, 4].values
y = pd.DataFrame(y)

print(f"\n[2] Features selected:")
print(f"    X (independent) shape : {X.shape}")
print(f"    y (dependent)   shape : {y.shape}")

# ============================================================
# Step 4 — Find null values in X
# ============================================================
print("\n[3] Null values BEFORE cleaning:")
print(X.isnull().sum().to_string())

# ============================================================
# Step 5 — Fill null values with column mean
# ============================================================
X = X.fillna(X.mean())

print("\n[4] Null values AFTER filling with mean:")
print(X.isnull().sum().to_string())
print("    ✓ All null values replaced with column mean")

# ============================================================
# Step 6 — Train / Test Split (80% train, 20% test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n[5] Train/Test Split (80/20):")
print(f"    Training samples : {X_train.shape[0]}")
print(f"    Test samples     : {X_test.shape[0]}")

# ============================================================
# Step 7 — Train the Linear Regression model
# ============================================================
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(f"\n[6] Model trained successfully (LinearRegression)")

# ============================================================
# Step 8 — Predict on test set
# ============================================================
y_pred = regressor.predict(X_test)

# ============================================================
# Step 9 — Evaluate model performance
# ============================================================
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n[7] Model Evaluation:")
print(f"    R² Score : {r2:.4f}   (1.0 = perfect fit)")
print(f"    MAE      : {mae:.4f}   (mean absolute error)")
print(f"    RMSE     : {rmse:.4f}   (root mean squared error)")

# ============================================================
# Step 10 — Plot 1: Actual vs. Predicted
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('HDI Score Predictor — Model Results', fontsize=14, fontweight='bold', y=1.01)

ax1 = axes[0]
min_val = float(min(y_test.min(), y_pred.min()))
max_val = float(max(y_test.max(), y_pred.max()))
ax1.scatter(y_test, y_pred, alpha=0.7, color='#2a78d6',
            edgecolors='white', linewidth=0.5, s=60, label='Predictions')
ax1.plot([min_val, max_val], [min_val, max_val],
         'r--', lw=1.5, label='Perfect fit (y = x)')
ax1.set_xlabel('Actual HDI Score', fontsize=11)
ax1.set_ylabel('Predicted HDI Score', fontsize=11)
ax1.set_title(f'Actual vs. Predicted HDI  (R² = {r2:.3f})', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)

# ============================================================
# Step 11 — Plot 2: Residuals
# ============================================================
ax2 = axes[1]
residuals = y_test.values.flatten() - y_pred.flatten()
ax2.scatter(y_pred, residuals, alpha=0.65, color='#1baf7a',
            edgecolors='white', linewidth=0.5, s=60)
ax2.axhline(y=0, color='red', linestyle='--', lw=1.5)
ax2.set_xlabel('Predicted HDI Score', fontsize=11)
ax2.set_ylabel('Residuals', fontsize=11)
ax2.set_title('Residuals Plot', fontsize=11)
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[8] Plots saved as 'model_results.png'")

# ============================================================
# Step 12 — Sample predictions table
# ============================================================
comparison = pd.DataFrame({
    'Actual HDI'    : y_test.values.flatten()[:10].round(4),
    'Predicted HDI' : y_pred.flatten()[:10].round(4),
    'Abs. Error'    : np.abs(y_test.values.flatten()[:10] - y_pred.flatten()[:10]).round(4)
})
print(f"\n[9] Sample predictions (first 10 test rows):")
print(comparison.to_string(index=False))
print("\n" + "=" * 55)
print("  Project complete!")
print("=" * 55)
