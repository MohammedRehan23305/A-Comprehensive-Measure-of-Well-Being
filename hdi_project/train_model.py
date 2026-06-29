"""
train_model.py
--------------
Trains a Linear Regression model to predict the Human Development Index (HDI)
and saves the trained model as HDI.pkl using Python's pickle module.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import pickle

# ── 1. Synthetic Dataset ───────────────────────────────────────────────────────
# HDI is influenced by:
#   - Life Expectancy Index  (LEI)
#   - Education Index        (EI)
#   - Income Index           (GNI)
np.random.seed(42)
n = 500

life_expectancy = np.random.uniform(40, 90, n)          # years (40–90)
mean_years_schooling = np.random.uniform(1, 15, n)      # years (1–15)
expected_years_schooling = np.random.uniform(5, 22, n)  # years (5–22)
gni_per_capita = np.random.uniform(500, 75000, n)       # USD (500–75 000)

# Normalise to 0-1 indices (simplified HDI methodology)
lei = (life_expectancy - 20) / (85 - 20)
mys = mean_years_schooling / 15
eys = expected_years_schooling / 18
ei  = (mys + eys) / 2
ii  = (np.log(gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100))

hdi = (lei * ei * ii) ** (1 / 3) + np.random.normal(0, 0.015, n)
hdi = np.clip(hdi, 0, 1)

df = pd.DataFrame({
    "Life Expectancy (years)":          life_expectancy,
    "Mean Years of Schooling":          mean_years_schooling,
    "Expected Years of Schooling":      expected_years_schooling,
    "GNI per Capita (USD)":            gni_per_capita,
    "HDI":                              hdi
})

# ── 2. Train / Test Split ──────────────────────────────────────────────────────
X = df[["Life Expectancy (years)",
        "Mean Years of Schooling",
        "Expected Years of Schooling",
        "GNI per Capita (USD)"]]
y = df["HDI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Train Model ─────────────────────────────────────────────────────────────
reg = LinearRegression()
reg.fit(X_train, y_train)

# ── 4. Evaluate ────────────────────────────────────────────────────────────────
y_pred = reg.predict(X_test)
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE     : {root_mean_squared_error(y_test, y_pred):.4f}")

# ── 5. Save Model with Pickle ──────────────────────────────────────────────────
#saving our model into a file
with open("HDI.pkl", "wb") as f:
    pickle.dump(reg, f)

print("✅  Model saved as HDI.pkl")
