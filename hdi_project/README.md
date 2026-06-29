# HDI Prediction — Multiple Linear Regression

## Project Overview
This project builds a Machine Learning pipeline to predict the
**Human Development Index (HDI)** score of a country using
socio-economic features.

## Features Used
| Feature                  | Description                          |
|--------------------------|--------------------------------------|
| Life Expectancy          | Average years a newborn is expected to live |
| Mean Years of Schooling  | Average years of education received  |
| Gross National Income    | GNI per capita (PPP, USD)            |
| Gender Development Index | Gender equality in human development |

## Target Variable
- **HDI** — Human Development Index (0 to 1 scale)

## Pipeline Steps
1. Load `Development.csv`
2. Handle null values (`isnull().sum()` → `fillna(mean)`)
3. Split features (X) and target (y)
4. 80/20 Train/Test split (`random_state=42`)
5. Instantiate & train `LinearRegression` from `sklearn`
6. Predict on test set
7. Evaluate with R², MAE, RMSE
8. Save visualisations to `model_results.png`

## Model Results
| Metric | Value  |
|--------|--------|
| R²     | 0.9688 |
| MAE    | 0.0135 |
| RMSE   | 0.0160 |

## Files
```
hdi_project/
├── Development.csv       # Dataset (64 countries)
├── hdi_prediction.py     # Main ML pipeline script
├── model_results.png     # Output visualisation
└── README.md             # This file
```

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib
python hdi_prediction.py
```
