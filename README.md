# HDI Score Predictor — Multiple Linear Regression

## Project Overview
Builds a **Multiple Linear Regression** model to predict the
**Human Development Index (HDI)** score using socio-economic indicators.

---

## Dataset Columns Used

### Independent Variables (X) — Model Inputs
| Index | Feature                            | Type        |
|-------|------------------------------------|-------------|
| 2     | Country                            | Categorical |
| 5     | Life Expectancy (years)            | Continuous  |
| 6     | Mean Years of Schooling            | Continuous  |
| 7     | Gross National Income (GNI)        | Continuous  |
| 67    | Internet Users / Gender Dev. Index | Continuous  |

### Dependent Variable (Y) — Prediction Target
| Index | Feature   | Range   |
|-------|-----------|---------|
| 4     | HDI Score | 0.0–1.0 |

---

## Full Pipeline (10 Steps)

| # | Step                        | Code                                      |
|---|-----------------------------|-------------------------------------------|
| 1 | Import libraries            | pandas, numpy, sklearn, matplotlib        |
| 2 | Load dataset                | `pd.read_csv('Development.csv')`          |
| 3 | Select X and y              | `iloc[:, [2,5,6,7,67]]` / `iloc[:, 4]`   |
| 4 | Find null values            | `X.isnull().sum()`                        |
| 5 | Fill nulls with mean        | `X = X.fillna(X.mean())`                 |
| 6 | Train / test split (80/20)  | `train_test_split(..., test_size=0.20)`   |
| 7 | Train model                 | `LinearRegression().fit(X_train, y_train)`|
| 8 | Predict                     | `regressor.predict(X_test)`               |
| 9 | Evaluate (R², MAE, RMSE)    | `r2_score`, `mean_absolute_error` etc.    |
|10 | Visualise & compare         | Scatter plot + Residuals + sample table   |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset in the same folder
#    (Development.csv must be present)

# 3. Run the script
python hdi_prediction.py
```

---

## Output
- Console output: null counts before/after, metrics, sample table
- `model_results.png` — Actual vs Predicted + Residuals (saved automatically)

---

## Expected Results
| Metric | Typical Value | Meaning                         |
|--------|---------------|---------------------------------|
| R²     | ~0.94         | 94% of variance explained       |
| MAE    | ~0.023        | Average error of ±0.023 points  |
| RMSE   | ~0.031        | Root mean squared error         |

---

## File Structure
```
hdi_project/
├── hdi_prediction.py   ← main script (all 10 steps)
├── Development.csv     ← your dataset (add manually)
├── requirements.txt    ← pip dependencies
└── README.md           ← this file
```
