# ============================================================
# Epic 2: Importing Required Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully.")

# ============================================================
# Epic 3: Dataset Download and Understanding
# ============================================================

# Story 1: Load the dataset
# Download from Kaggle: human-development-index dataset
# Place 'hdi.csv' inside the datasets/ folder before running

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'hdi.csv')

# --- Synthetic dataset generator (fallback if CSV not found) ---
def generate_synthetic_hdi_data(n=500, seed=42):
    np.random.seed(seed)
    countries = ['India', 'USA', 'Germany', 'Brazil', 'China', 'Nigeria',
                 'France', 'Japan', 'Australia', 'Mexico']
    regions   = ['Asia', 'Americas', 'Europe', 'Africa', 'Oceania']

    data = {
        'Country':                np.random.choice(countries, n),
        'Region':                 np.random.choice(regions, n),
        'Life_Expectancy':        np.round(np.random.uniform(50, 85, n), 2),
        'Mean_Years_Schooling':   np.round(np.random.uniform(3, 15, n), 2),
        'Expected_Years_Schooling': np.round(np.random.uniform(5, 20, n), 2),
        'GNI_Per_Capita':         np.round(np.random.uniform(500, 80000, n), 2),
        'Population':             np.random.randint(100000, 1400000000, n),
        'Internet_Usage_Pct':     np.round(np.random.uniform(5, 98, n), 2),
        'CO2_Emissions':          np.round(np.random.uniform(0.1, 15, n), 2),
    }
    df = pd.DataFrame(data)
    # HDI formula (simplified approximation)
    life_index     = (df['Life_Expectancy'] - 20) / (85 - 20)
    edu_index      = (0.5 * (df['Mean_Years_Schooling'] / 15) +
                      0.5 * (df['Expected_Years_Schooling'] / 18))
    income_index   = (np.log(df['GNI_Per_Capita']) - np.log(100)) / (np.log(75000) - np.log(100))
    df['HDI'] = np.round((life_index * edu_index * income_index) ** (1/3), 4).clip(0.1, 0.99)
    return df

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded from {DATASET_PATH}")
except FileNotFoundError:
    print("⚠️  hdi.csv not found — using synthetic dataset for demonstration.")
    df = generate_synthetic_hdi_data()

# Story 2: Explore dataset structure
print("\n========== Dataset Overview ==========")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())

# Story 3: Data Visualization
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'static', 'plots'), exist_ok=True)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'plots')

def save_plot(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Saved: {name}")

print("\n📊 Generating visualizations...")

# 1. HDI Distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df['HDI'], kde=True, color='steelblue', ax=ax)
ax.set_title('Distribution of HDI Values')
ax.set_xlabel('HDI')
save_plot(fig, 'hdi_distribution.png')

# 2. Correlation heatmap (numeric columns only)
fig, ax = plt.subplots(figsize=(10, 7))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
ax.set_title('Feature Correlation Heatmap')
save_plot(fig, 'correlation_heatmap.png')

# 3. Scatter: GNI vs HDI
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['GNI_Per_Capita'], df['HDI'], alpha=0.5, color='teal')
ax.set_xlabel('GNI Per Capita')
ax.set_ylabel('HDI')
ax.set_title('GNI Per Capita vs HDI')
save_plot(fig, 'gni_vs_hdi.png')

# 4. Life Expectancy vs HDI
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['Life_Expectancy'], df['HDI'], alpha=0.5, color='salmon')
ax.set_xlabel('Life Expectancy')
ax.set_ylabel('HDI')
ax.set_title('Life Expectancy vs HDI')
save_plot(fig, 'life_exp_vs_hdi.png')

# ============================================================
# Epic 4: Data Preprocessing and Label Encoding
# ============================================================

print("\n========== Preprocessing ==========")

# Story 2: Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)
print(f"Shape after dropping NaN rows: {df.shape}")

# Story 3: Label Encoding for categorical columns
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Encoding columns: {categorical_cols}")
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Story 1 & 4: Select features and target
TARGET   = 'HDI'
FEATURES = [c for c in df.columns if c != TARGET]
print(f"Features: {FEATURES}")
print(f"Target  : {TARGET}")

X = df[FEATURES]
y = df[TARGET]

# ============================================================
# Epic 5: Train / Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ============================================================
# Epic 6: Fitting the Model
# ============================================================
print("\n========== Model Training ==========")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Linear Regression model trained.")

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n📈 Model Performance:")
print(f"  MAE  : {mae:.4f}")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# Actual vs Predicted plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_test, y_pred, alpha=0.5, color='purple')
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual HDI')
ax.set_ylabel('Predicted HDI')
ax.set_title('Actual vs Predicted HDI')
save_plot(fig, 'actual_vs_predicted.png')

# ============================================================
# Epic 7: Saving the Model
# ============================================================
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

model_path    = os.path.join(MODELS_DIR, 'hdi_model.pkl')
features_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(features_path, 'wb') as f:
    pickle.dump(FEATURES, f)

print(f"\n✅ Model saved  → {model_path}")
print(f"✅ Features saved → {features_path}")
print("\n🎉 Training pipeline complete! Run app.py to launch the web app.")
