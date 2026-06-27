# HDI Score Predictor — Multiple Linear Regression

## Project Overview
This project builds a **Multiple Linear Regression** model to predict
the **Human Development Index (HDI)** score for countries, using
socio-economic indicators from the Development dataset.

---

## Dataset Columns Used

### Independent Variables (X) — Model Inputs
| Index | Feature                   | Type        |
|-------|---------------------------|-------------|
| 2     | Country                   | Categorical |
| 5     | Life Expectancy (years)   | Continuous  |
| 6     | Mean Years of Schooling   | Continuous  |
| 7     | Gross National Income     | Continuous  |
| 67    | Gender Development Index  | Continuous  |

### Dependent Variable (Y) — Prediction Target
| Index | Feature   | Range   |
|-------|-----------|---------|
| 4     | HDI Score | 0.0–1.0 |

---

## Project Steps

1. **Import libraries** — pandas, numpy, scikit-learn, matplotlib
2. **Load dataset** — `Development.csv`
3. **Select features** — X (indices 2,5,6,7,67) and y (index 4)
4. **Train/test split** — 80% train, 20% test (`random_state=42`)
5. **Train model** — `LinearRegression().fit(X_train, y_train)`
6. **Predict** — `regressor.predict(X_test)`
7. **Evaluate** — R², MAE, RMSE
8. **Visualise** — Actual vs. Predicted scatter plot + Residuals plot
9. **Compare** — Sample prediction table

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset in the same folder
#    Make sure Development.csv is present

# 3. Run the script
python hdi_prediction.py
```

---

## Output Files Generated
- `actual_vs_predicted.png` — scatter plot of model accuracy
- `residuals.png` — residuals distribution plot

---

## Expected Results
| Metric | Typical Value | Meaning                        |
|--------|---------------|--------------------------------|
| R²     | ~0.94         | 94% of variance explained      |
| MAE    | ~0.023        | Average error of ±0.023 points |
| RMSE   | ~0.031        | Root mean squared error        |

---

## File Structure
```
hdi_project/
├── hdi_prediction.py   ← main script
├── Development.csv     ← dataset (add your own)
├── requirements.txt    ← Python dependencies
└── README.md           ← this file
```
