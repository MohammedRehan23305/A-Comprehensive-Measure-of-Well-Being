# Human Development Index (HDI) Prediction App

A simple Flask web application that predicts a country's HDI category
(Low / Medium / High / Very High) from four inputs: Country code,
Life expectancy, Mean years of schooling, and GNI per capita.

## Project structure
```
hdi_app/
├── app.py              # Flask application (routes + prediction logic)
├── train_model.py       # Script used to train and save HDI.pkl
├── HDI.pkl               # Trained model (already included)
├── requirements.txt
└── templates/
    ├── home.html          # Landing page
    ├── indexnew.html       # Input form
    └── resultnew.html       # Prediction result page
```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Retrain the model on your own dataset:
   ```
   python train_model.py
   ```
   This script currently trains on synthetic data as a placeholder —
   replace the data-loading section with your real HDI dataset
   (e.g. from the UNDP Human Development Reports) for accurate predictions.

4. Run the app:
   ```
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000/`.

## Notes
- The included `HDI.pkl` is trained on synthetic data so the app runs
  out of the box. Swap in your real trained model file (same filename
  and same 4 input features, in the same order) for production use.
- HDI category thresholds used in `app.py`:
  - 0.30 – 0.40 → Low HDI
  - 0.40 – 0.70 → Medium HDI
  - 0.70 – 0.80 → High HDI
  - 0.80 – 0.94 → Very High HDI
  - outside this range → "values do not match any HDI category"
