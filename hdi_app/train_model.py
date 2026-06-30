# train_model.py
# Generates a synthetic dataset and trains a simple regression model
# to predict the Human Development Index (HDI), then saves it as HDI.pkl
# Replace this with your real training pipeline / dataset if you have one.

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

np.random.seed(42)

n_samples = 500

# Synthetic features
country_code = np.random.randint(1, 195, n_samples).astype(float)          # encoded country id
life_expectancy = np.random.uniform(50, 85, n_samples)                     # years
mean_years_schooling = np.random.uniform(2, 14, n_samples)                 # years
gni_per_capita = np.random.uniform(500, 80000, n_samples)                  # USD

# Synthetic target: a weighted combination, scaled to roughly [0, 1]
hdi = (
    0.25 * (life_expectancy / 85)
    + 0.35 * (mean_years_schooling / 14)
    + 0.40 * (np.log(gni_per_capita) / np.log(80000))
)
hdi = np.clip(hdi + np.random.normal(0, 0.02, n_samples), 0.2, 0.98)

features_name = [
    'Country',
    'Life expectancy',
    'Mean years of schooling',
    'Gross national income (GNI) per capita'
]

df = pd.DataFrame(
    {
        'Country': country_code,
        'Life expectancy': life_expectancy,
        'Mean years of schooling': mean_years_schooling,
        'Gross national income (GNI) per capita': gni_per_capita,
    },
    columns=features_name
)

model = LinearRegression()
model.fit(df, hdi)

with open('HDI.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as HDI.pkl")
print("Sample prediction:", model.predict(df.iloc[:1]))
