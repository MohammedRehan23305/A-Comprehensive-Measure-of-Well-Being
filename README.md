# 🌍 HDI Prediction — Linear Regression + Flask

## Project Structure
```
hdi_prediction/
├── app.py                  # Epic 8 — Flask web application
├── requirements.txt        # Epic 1 — dependencies
├── datasets/               # Place hdi.csv here (from Kaggle)
├── models/                 # Auto-created: saved model + features
├── src/
│   └── train.py            # Epics 2-7 — full ML pipeline
├── templates/
│   ├── index.html          # Prediction form
│   ├── result.html         # Result page
│   └── plots.html          # EDA visualizations
└── static/
    ├── css/style.css
    └── plots/              # Auto-created: EDA plot PNGs
```

## ▶️ How to Run

### Step 1 — Install dependencies (Epic 1)
```bash
pip install -r requirements.txt
```

### Step 2 — (Optional) Add your Kaggle dataset
Download an HDI dataset from Kaggle and place the CSV at:
```
datasets/hdi.csv
```
> If not provided, the training script auto-generates a synthetic dataset.

### Step 3 — Train the model (Epics 2–7)
```bash
python src/train.py
```
This will:
- Load/generate data
- Visualize and explore (EDA plots saved to static/plots/)
- Preprocess and label-encode
- Train Linear Regression
- Evaluate and print metrics (MAE, MSE, RMSE, R²)
- Save model → models/hdi_model.pkl

### Step 4 — Launch the Flask app (Epic 8)
```bash
python app.py
```
Open your browser at: **http://127.0.0.1:5000**

## 🔌 JSON API
```bash
POST /api/predict
Content-Type: application/json

{
  "Life_Expectancy": 72,
  "Mean_Years_Schooling": 10,
  "Expected_Years_Schooling": 14,
  "GNI_Per_Capita": 15000,
  "Population": 1000000,
  "Internet_Usage_Pct": 60,
  "CO2_Emissions": 3.5,
  "Country": 3,
  "Region": 1
}
```

## 📊 HDI Bands
| Score     | Category          |
|-----------|-------------------|
| ≥ 0.800   | Very High         |
| 0.700–0.799 | High            |
| 0.550–0.699 | Medium          |
| < 0.550   | Low               |
