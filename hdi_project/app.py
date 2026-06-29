"""
app.py
------
Flask web application that loads the saved HDI model (HDI.pkl)
and serves predictions via a simple HTML form.
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load the saved model once at startup ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "HDI.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("✅  HDI.pkl loaded successfully.")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Render the prediction form."""
    return render_template("index.html", prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    """Accept form data, run the model, return the HDI prediction."""
    prediction = None
    error = None

    try:
        life_exp   = float(request.form["life_expectancy"])
        mys        = float(request.form["mean_years_schooling"])
        eys        = float(request.form["expected_years_schooling"])
        gni        = float(request.form["gni_per_capita"])

        # Basic validation
        if not (20 <= life_exp <= 90):
            raise ValueError("Life expectancy must be between 20 and 90.")
        if not (0 <= mys <= 15):
            raise ValueError("Mean years of schooling must be between 0 and 15.")
        if not (0 <= eys <= 22):
            raise ValueError("Expected years of schooling must be between 0 and 22.")
        if not (100 <= gni <= 200000):
            raise ValueError("GNI per capita must be between 100 and 200,000.")

        features = np.array([[life_exp, mys, eys, gni]])
        raw = model.predict(features)[0]
        prediction = round(float(np.clip(raw, 0, 1)), 4)

    except ValueError as ve:
        error = str(ve)
    except Exception as ex:
        error = f"Unexpected error: {ex}"

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)
