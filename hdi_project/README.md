# HDI Prediction — Multiple Linear Regression

## Project Overview
Predicts **Human Development Index (HDI)** scores from country-level
socio-economic indicators using Multiple Linear Regression (scikit-learn).

## Features Used
| Feature                  | Description                            |
|--------------------------|----------------------------------------|
| Life Expectancy          | Average lifespan at birth (years)      |
| Mean Years of Schooling  | Average education years for adults     |
| Gross National Income    | GNI per capita (PPP, USD)              |
| Gender Development Index | Gender equality measure (0–1 scale)    |

## Pipeline Steps
1. **Load data** — `Development.csv` (69 countries)
2. **Null handling** — `isnull().sum()` check + `fillna(mean())`
3. **Train/Test split** — 80/20 (`random_state=42`)
4. **Train** — `LinearRegression` from scikit-learn
5. **Predict** — `reg.predict(x_test)` → `y_pred`
6. **Evaluate** — R², MAE, RMSE
7. **Test with fewer values** — predict for 3 individual sample inputs
8. **Compare** — `y_test` vs `y_pred` side-by-side table
9. **Visualise** — `model_results.png` (4 plots)

## Results
| Metric | Value  |
|--------|--------|
| R²     | 0.9741 |
| MAE    | 0.0131 |
| RMSE   | 0.0148 |

The model explains **97.4 %** of variance in HDI scores.

## Files
```
hdi_project/
├── hdi_prediction.py   ← main script
├── Development.csv     ← dataset (69 countries, 6 columns)
├── model_results.png   ← 4-panel visualisation
└── README.md
```

## Run
```bash
pip install pandas numpy scikit-learn matplotlib
python hdi_prediction.py
```
