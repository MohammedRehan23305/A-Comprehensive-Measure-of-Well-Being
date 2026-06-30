# Human Development Index (HDI) Prediction App

A Flask web application that predicts a country's HDI score and category
(Low / Medium / High / Very High) from five inputs: Country, Life expectancy,
Mean years of schooling, GNI per capita, and Internet users.

## Project structure
```
hdi_app/
├── app.py                # Flask application (routes + prediction logic)
├── train_model.py        # Script used to train and save HDI.pkl
├── HDI.pkl                # Trained model (already included)
├── requirements.txt
└── templates/
    ├── home.html           # Landing page (intro + Predict/Home navbar)
    ├── indexnew.html        # Prediction form (country dropdown + 4 inputs)
    └── resultnew.html        # Result page (HDI score + category)
```

## Pages
- **Home Page (`home.html`)** — introduces the HDI and its significance, with
  a navbar (Predict / Home) in the top-right corner.
- **Prediction Page (`indexnew.html`)** — lets the user select a country from
  a dropdown and enter Life expectancy (50-89), Mean years of schooling
  (1.1-15), GNI per capita (740-140000), and Internet users, then submits via
  the "Predict" button.
- **Result Page (`resultnew.html`)** — displays the predicted HDI score and
  category (Low / Medium / High / Very High HDI).

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
   This script currently trains on synthetic data as a placeholder — replace
   the data-loading section with your real HDI dataset (e.g. from the UNDP
   Human Development Reports) for accurate predictions. Keep the same 5
   feature names/order if you do.

4. Run the app:
   ```
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000/`.

## Notes
- The included `HDI.pkl` is trained on synthetic data so the app runs out of
  the box. Swap in your real trained model file (same filename, same 5 input
  features in the same order) for production use.
- HDI category thresholds used in `app.py`:
  - 0.30 – 0.40 → Low HDI
  - 0.40 – 0.70 → Medium HDI
  - 0.70 – 0.80 → High HDI
  - 0.80 – 0.94 → Very High HDI
  - outside this range → "values do not match any HDI category"
- The country dropdown currently lists a sample set of countries (Afghanistan,
  Australia, Bangladesh, Canada, India, Poland, Turkey) with their encoded
  values — add more `<option>` entries in `indexnew.html` to match your full
  training dataset's country encoding.
