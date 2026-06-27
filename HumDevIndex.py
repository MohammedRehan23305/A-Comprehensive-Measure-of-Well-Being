# ============================================================
#  ML - 0027 | Human Development Index Prediction
#  File   : HumDevIndex.py
#  Author : ML Learner
#  Dataset: Dataset/HDI.csv
# ============================================================

# ─────────────────────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error,
                              r2_score)

print("✔ All libraries imported successfully.\n")


# ─────────────────────────────────────────────────────────────
# STEP 2: LOAD DATASET
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('Dataset/HDI.csv')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe())
print("\nMissing values per column:")
print(df.isnull().sum())


# ─────────────────────────────────────────────────────────────
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────

# 3a. HDI Tier distribution — bar chart
plt.figure(figsize=(8, 5))
tier_counts = df['HDI_Tier'].value_counts()
colors = ['#1baf7a', '#2a78d6', '#eda100', '#e34948']
sns.barplot(x=tier_counts.index, y=tier_counts.values, palette=colors)
plt.title('HDI Tier Distribution (191 countries)')
plt.xlabel('Tier')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('plots/tier_distribution.png', dpi=150)
plt.show()

# 3b. HDI value distribution — histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['HDI'], bins=25, kde=True, color='#2a78d6')
plt.title('Distribution of HDI Values')
plt.xlabel('HDI Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/hdi_distribution.png', dpi=150)
plt.show()

# 3c. Strip plot — HDI vs Tier
plt.figure(figsize=(9, 5))
order = ['Low', 'Medium', 'High', 'Very High']
sns.stripplot(data=df, x='HDI_Tier', y='HDI',
              order=order, jitter=True,
              palette=['#e34948', '#eda100', '#2a78d6', '#1baf7a'],
              size=5, alpha=0.7)
plt.title('HDI Score by Development Tier (Strip Plot)')
plt.xlabel('HDI Tier')
plt.ylabel('HDI Score')
plt.tight_layout()
plt.savefig('plots/strip_plot.png', dpi=150)
plt.show()

# 3d. Correlation heatmap
plt.figure(figsize=(9, 7))
numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5,
            annot_kws={'size': 10})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=150)
plt.show()

# 3e. Pairplot of key features
key_features = ['HDI', 'Life_Expectancy',
                'Expected_School_Years', 'Mean_School_Years',
                'GNI_per_Capita']
sns.pairplot(df[key_features], diag_kind='kde',
             plot_kws={'alpha': 0.5, 'color': '#2a78d6'})
plt.suptitle('Pairplot of Key HDI Features', y=1.02)
plt.tight_layout()
plt.savefig('plots/pairplot.png', dpi=150)
plt.show()

print("✔ EDA complete — plots saved to plots/\n")


# ─────────────────────────────────────────────────────────────
# STEP 4: PREPROCESSING
# ─────────────────────────────────────────────────────────────

# 4a. Drop missing values
df.dropna(inplace=True)
print(f"Rows after dropping nulls: {len(df)}")

# 4b. Encode HDI_Tier → numeric
le = LabelEncoder()
df['HDI_Tier_Encoded'] = le.fit_transform(df['HDI_Tier'])
# Mapping: Low=0, Medium=1, High=2, Very High=3
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4c. Log-transform GNI to reduce right skew
df['GNI_log'] = np.log1p(df['GNI_per_Capita'])

# 4d. Define features and target
features = ['Life_Expectancy',
            'Expected_School_Years',
            'Mean_School_Years',
            'GNI_log',
            'HDI_Tier_Encoded']

X = df[features]
y = df['HDI']

# 4e. Train / test split  (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\nTrain size : {len(X_train)}")
print(f"Test  size : {len(X_test)}")

# 4f. Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("✔ Preprocessing complete.\n")


# ─────────────────────────────────────────────────────────────
# STEP 5: MODEL TRAINING
# ─────────────────────────────────────────────────────────────

# ── Model 1: Linear Regression (baseline) ──────────────────
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
print("Linear Regression trained.")

# ── Model 2: Random Forest Regressor ───────────────────────
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ── Model 3: Gradient Boosting Regressor (best) ────────────
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.85,
    random_state=42
)
gb.fit(X_train, y_train)
print("Gradient Boosting trained.\n")


# ─────────────────────────────────────────────────────────────
# STEP 6: MODEL EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Print R², MAE, RMSE, and 5-fold CV R² for a model."""
    pred = model.predict(X_te)
    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    cv   = cross_val_score(model, X_tr, y_tr,
                           cv=5, scoring='r2').mean()
    print(f"{'─'*50}")
    print(f"Model   : {name}")
    print(f"R²      : {r2:.4f}")
    print(f"MAE     : {mae:.4f}")
    print(f"RMSE    : {rmse:.4f}")
    print(f"CV R²   : {cv:.4f}  (5-fold)")
    return pred

lr_pred = evaluate_model("Linear Regression",
                          lr, X_train_sc, y_train,
                          X_test_sc,  y_test)

rf_pred = evaluate_model("Random Forest",
                          rf, X_train, y_train,
                          X_test, y_test)

gb_pred = evaluate_model("Gradient Boosting (Best)",
                          gb, X_train, y_train,
                          X_test, y_test)
print('─'*50)

# 6a. Actual vs Predicted scatter — Gradient Boosting
plt.figure(figsize=(7, 6))
plt.scatter(y_test, gb_pred, alpha=0.7, color='#2a78d6', edgecolors='white', s=60)
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
plt.xlabel('Actual HDI')
plt.ylabel('Predicted HDI')
plt.title('Actual vs Predicted HDI (Gradient Boosting)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150)
plt.show()

# 6b. Residual plot
residuals = y_test.values - gb_pred
plt.figure(figsize=(8, 5))
plt.scatter(gb_pred, residuals, alpha=0.6, color='#eda100', edgecolors='white', s=55)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted HDI')
plt.ylabel('Residual')
plt.title('Residual Plot — Gradient Boosting')
plt.tight_layout()
plt.savefig('plots/residuals.png', dpi=150)
plt.show()

# 6c. Feature importance — Random Forest
importances = rf.feature_importances_
fi_df = pd.DataFrame({'Feature': features,
                       'Importance': importances}
                     ).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=fi_df, x='Importance', y='Feature',
            palette='Blues_r')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()

print("\nFeature Importances:")
print(fi_df.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# STEP 7: SAVE MODEL (for Flask app)
# ─────────────────────────────────────────────────────────────
with open('Flask/HDI.pkl', 'wb') as f:
    pickle.dump(gb, f)

# Also save scaler for inference
with open('Flask/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✔ Model saved → Flask/HDI.pkl")
print("✔ Scaler saved → Flask/scaler.pkl")


# ─────────────────────────────────────────────────────────────
# STEP 8: MAKE A SAMPLE PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_hdi(life_exp, exp_school, mean_school, gni, tier_enc):
    """
    Predict HDI for given inputs.

    Parameters
    ----------
    life_exp    : float  — life expectancy in years
    exp_school  : float  — expected years of schooling
    mean_school : float  — mean years of schooling
    gni         : float  — GNI per capita (PPP $)
    tier_enc    : int    — 0=Low, 1=Medium, 2=High, 3=Very High

    Returns
    -------
    hdi   : float
    label : str
    """
    gni_log = np.log1p(gni)
    X_in = np.array([[life_exp, exp_school, mean_school,
                       gni_log, tier_enc]])
    hdi = float(gb.predict(X_in)[0])
    if hdi >= 0.80:
        label = 'Very High Development'
    elif hdi >= 0.70:
        label = 'High Development'
    elif hdi >= 0.55:
        label = 'Medium Development'
    else:
        label = 'Low Development'
    return round(hdi, 3), label


# Example — India-like values
hdi_val, hdi_label = predict_hdi(
    life_exp=70.2,
    exp_school=11.9,
    mean_school=6.7,
    gni=6590,
    tier_enc=1          # Medium
)
print(f"\nSample Prediction:")
print(f"  HDI Score : {hdi_val}")
print(f"  Tier      : {hdi_label}")

print("\n✔ HumDevIndex.py completed successfully.")
