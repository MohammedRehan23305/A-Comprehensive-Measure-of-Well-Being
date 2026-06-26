# ============================================================
# Epic 8: Building the Flask Web Application
# Story 1: Flask backend — handle requests, load model, predict
# ============================================================
import os
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)

# --- Load model & feature list ---
MODELS_DIR    = os.path.join(os.path.dirname(__file__), 'models')
model_path    = os.path.join(MODELS_DIR, 'hdi_model.pkl')
features_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(features_path, 'rb') as f:
        FEATURES = pickle.load(f)
    print(f"✅ Model loaded. Features: {FEATURES}")
except FileNotFoundError:
    raise RuntimeError(
        "Model files not found. Please run  src/train.py  first to train and save the model."
    )

# ============================================================
# Routes
# ============================================================

@app.route('/')
def home():
    """Render the prediction form."""
    return render_template('index.html', features=FEATURES)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept form data, run prediction, return result."""
    try:
        values = []
        for feat in FEATURES:
            raw = request.form.get(feat, 0)
            values.append(float(raw))

        input_array = np.array(values).reshape(1, -1)
        prediction  = model.predict(input_array)[0]
        prediction  = round(float(prediction), 4)

        # HDI band label
        if prediction >= 0.8:
            band = ("Very High", "success")
        elif prediction >= 0.7:
            band = ("High", "info")
        elif prediction >= 0.55:
            band = ("Medium", "warning")
        else:
            band = ("Low", "danger")

        return render_template(
            'result.html',
            prediction=prediction,
            band_label=band[0],
            band_class=band[1],
            features=FEATURES,
            values=dict(zip(FEATURES, values)),
        )

    except Exception as e:
        return render_template('index.html', features=FEATURES,
                               error=f"Prediction failed: {str(e)}")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic access."""
    try:
        data   = request.get_json(force=True)
        values = [float(data.get(f, 0)) for f in FEATURES]
        pred   = round(float(model.predict([values])[0]), 4)
        return jsonify({'hdi_prediction': pred, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400


@app.route('/plots')
def plots():
    """Show EDA visualizations."""
    plots_dir  = os.path.join(app.static_folder, 'plots')
    plot_files = []
    if os.path.isdir(plots_dir):
        plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
    return render_template('plots.html', plot_files=plot_files)


# ============================================================
# Story 3: Run and validate the application
# ============================================================
if __name__ == '__main__':
    print("\n🚀 Starting HDI Prediction Flask App...")
    print("   Visit: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
