from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model bundle (model + scaler)
with open('HDI.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']


def classify_hdi(hdi):
    """Classify HDI into development categories."""
    if hdi >= 0.800:
        return 'Very High Human Development', '#27ae60'
    elif hdi >= 0.700:
        return 'High Human Development', '#2980b9'
    elif hdi >= 0.550:
        return 'Medium Human Development', '#f39c12'
    else:
        return 'Low Human Development', '#e74c3c'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        life_exp = float(request.form['life_expectancy'])
        edu_index = float(request.form['education_index'])
        gni = float(request.form['gni_per_capita'])

        # Validate ranges
        if not (20 <= life_exp <= 90):
            return render_template('index.html', error='Life Expectancy must be between 20 and 90 years.')
        if not (0.0 <= edu_index <= 1.0):
            return render_template('index.html', error='Education Index must be between 0.0 and 1.0.')
        if not (100 <= gni <= 150000):
            return render_template('index.html', error='GNI per Capita must be between $100 and $150,000.')

        features = pd.DataFrame([[life_exp, edu_index, gni]],
                                columns=['Life_Expectancy', 'Education_Index', 'GNI_per_Capita'])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        prediction = round(float(prediction), 3)

        category, color = classify_hdi(prediction)

        return render_template(
            'index.html',
            prediction=prediction,
            category=category,
            category_color=color,
            life_exp=life_exp,
            edu_index=edu_index,
            gni=gni
        )
    except ValueError:
        return render_template('index.html', error='Please enter valid numeric values in all fields.')
    except Exception as e:
        return render_template('index.html', error=f'Prediction error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
