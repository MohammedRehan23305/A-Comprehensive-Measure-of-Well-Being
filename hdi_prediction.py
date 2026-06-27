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
print("Dataset loaded successfully!")
print(f"Shape: {Development.shape}")
print(f"\nFirst 5 rows:\n{Development.head()}")
print(f"\nColumn names:\n{list(Development.columns)}")

# ============================================================
# Step 3 — Select features (X) and target (y)
# ============================================================

# Independent Variables (X): Inputs given to the model
# Index 2  → Country
# Index 5  → Life Expectancy
# Index 6  → Mean Years of Schooling
# Index 7  → Gross National Income (GNI)
# Index 67 → Gender Development Index
X = Development.iloc[:, [2, 5, 6, 7, 67]]
X = pd.DataFrame(X)

# Dependent Variable (y): HDI Score (column index 4)
y = Development.iloc[:, 4].values
y = pd.DataFrame(y)

print("\n--- Feature Selection ---")
print(f"Independent Variables (X) shape: {X.shape}")
print(f"X columns: {list(X.columns)}")
print(f"\nDependent Variable (y) shape: {y.shape}")
print(f"y column: {list(y.columns)}")

# ============================================================
# Step 4 — Train / Test Split (80% train, 20% test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n--- Train/Test Split ---")
print(f"Training set size : {X_train.shape[0]} samples")
print(f"Test set size     : {X_test.shape[0]} samples")

# ============================================================
# Step 5 — Train the Linear Regression model
# ============================================================
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("\n--- Model Trained ---")
print(f"Intercept    : {regressor.intercept_}")
print(f"Coefficients : {regressor.coef_}")

# ============================================================
# Step 6 — Predict on test set
# ============================================================
y_pred = regressor.predict(X_test)

# ============================================================
# Step 7 — Evaluate model performance
# ============================================================
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Evaluation ---")
print(f"R² Score : {r2:.4f}  (1.0 = perfect fit)")
print(f"MAE      : {mae:.4f}  (mean absolute error)")
print(f"RMSE     : {rmse:.4f}  (root mean squared error)")

# ============================================================
# Step 8 — Visualise: Actual vs. Predicted HDI scores
# ============================================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='#2a78d6', edgecolors='white', linewidth=0.5, s=60, label='Predictions')
min_val = float(min(y_test.min(), y_pred.min()))
max_val = float(max(y_test.max(), y_pred.max()))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Perfect fit')
plt.xlabel('Actual HDI Score', fontsize=12)
plt.ylabel('Predicted HDI Score', fontsize=12)
plt.title('Actual vs. Predicted HDI Scores\nLinear Regression Model', fontsize=13, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()
print("\nPlot saved as 'actual_vs_predicted.png'")

# ============================================================
# Step 9 — Visualise: Residuals plot
# ============================================================
residuals = y_test.values.flatten() - y_pred.flatten()

plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.6, color='#1baf7a', edgecolors='white', linewidth=0.5)
plt.axhline(y=0, color='red', linestyle='--', lw=1.5)
plt.xlabel('Predicted HDI Score', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals Plot', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
plt.show()
print("Residuals plot saved as 'residuals.png'")

# ============================================================
# Step 10 — Sample predictions comparison
# ============================================================
comparison = pd.DataFrame({
    'Actual HDI'   : y_test.values.flatten()[:10],
    'Predicted HDI': y_pred.flatten()[:10],
    'Difference'   : abs(y_test.values.flatten()[:10] - y_pred.flatten()[:10])
})
comparison = comparison.round(4)
print("\n--- Sample Predictions (first 10 test rows) ---")
print(comparison.to_string(index=False))
