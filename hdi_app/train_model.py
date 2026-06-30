# train_model.py
# Generates a synthetic dataset and trains a regression model to predict
# the Human Development Index (HDI), then saves it as HDI.pkl
# Replace the data-loading section with your real dataset for accurate predictions.

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

np.random.seed(42)

n_samples = 800

# Synthetic features, matching the ranges shown on the prediction form
country_code = np.random.randint(0, 195, n_samples).astype(float)         # encoded country id
life_expectancy = np.random.uniform(50, 89, n_samples)                    # years (range 50-89)
mean_years_schooling = np.random.uniform(1.1, 15, n_samples)              # years (range 1.1-15)
gni_per_capita = np.random.uniform(740, 140000, n_samples)                # USD  (range 740-140000)
internet_users = np.random.uniform(0, 100, n_samples)                     # % of population

# Synthetic target: weighted combination, scaled to roughly [0, 1]
hdi = (
    0.22 * (life_expectancy / 89)
    + 0.30 * (mean_years_schooling / 15)
    + 0.30 * (np.log(gni_per_capita) / np.log(140000))
    + 0.18 * (internet_users / 100)
)
hdi = np.clip(hdi + np.random.normal(0, 0.02, n_samples), 0.2, 0.98)

features_name = [
    'Country',
    'Life expectancy',
    'Mean years of schooling',
    'Gross national income (GNI) per capita',
    'Internet users'
]

df = pd.DataFrame(
    {
        'Country': country_code,
        'Life expectancy': life_expectancy,
        'Mean years of schooling': mean_years_schooling,
        'Gross national income (GNI) per capita': gni_per_capita,
        'Internet users': internet_users,
    },
    columns=features_name
)

model = LinearRegression()
model.fit(df, hdi)

with open('HDI.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as HDI.pkl")
print("Sample prediction:", model.predict(df.iloc[:1]))
