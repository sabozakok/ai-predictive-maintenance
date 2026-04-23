import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / 'models'
MODEL_PATH = MODELS_DIR / 'predictive_model.joblib'
FEATURES_PATH = MODELS_DIR / 'feature_columns.json'

app = Flask(__name__)

EXPECTED_RANGES = {
    'temperature': (0, 200),
    'vibration': (0, 20),
    'pressure': (0, 200),
    'rpm': (0, 10000),
    'runtime_hours': (0, 100000),
    'load_percent': (0, 150),
    'oil_quality_index': (0, 100),
}


class ModelService:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.load()

    def load(self):
        if MODEL_PATH.exists() and FEATURES_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
                self.feature_columns = json.load(f)

    @property
    def is_ready(self):
        return self.model is not None and bool(self.feature_columns)

    def predict(self, payload: dict) -> dict:
        if not self.is_ready:
            raise RuntimeError('Model is not ready. Train the model first.')

        data = {}
        missing = []
        for col in self.feature_columns:
            if col not in payload:
                missing.append(col)
                continue
            data[col] = float(payload[col])

        if missing:
            raise ValueError(f'Missing required fields: {", ".join(missing)}')

        for key, value in data.items():
            low, high = EXPECTED_RANGES[key]
            if not (low <= value <= high):
                raise ValueError(f'Field {key}={value} is outside expected range {low}-{high}')

        X = pd.DataFrame([data], columns=self.feature_columns)
        probability = float(self.model.predict_proba(X)[0][1])
        prediction = int(probability >= 0.55)

        if probability < 0.35:
            risk_level = 'Low'
            recommendation = 'Continue routine monitoring and scheduled maintenance.'
        elif probability < 0.65:
            risk_level = 'Medium'
            recommendation = 'Inspect equipment soon and review operating conditions.'
        else:
            risk_level = 'High'
            recommendation = 'Plan urgent inspection and preventive maintenance.'

        return {
            'failure_prediction': prediction,
            'failure_probability': round(probability, 4),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'input': data,
        }


service = ModelService()


@app.route('/')
def index():
    return render_template('index.html', model_ready=service.is_ready)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_ready': service.is_ready,
        'model_path': str(MODEL_PATH),
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.get_json(force=True)
        result = service.predict(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
