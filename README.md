# AI Predictive Maintenance System

An end-to-end starter project for industrial predictive maintenance using Python, scikit-learn, Flask, and a lightweight web dashboard.

## What this project includes
- Synthetic industrial equipment dataset generator
- Model training pipeline for failure-risk prediction
- Flask REST API for real-time inference
- Web dashboard for manual prediction and sample monitoring
- Docker support for easy deployment
- Clean structure suitable for GitHub portfolio use

## Use case
This project is designed as a portfolio-ready example for oil & gas, power generation, utilities, and industrial operations. It simulates equipment telemetry such as:
- Temperature
- Vibration
- Pressure
- RPM
- Runtime hours
- Load percentage
- Oil quality index

The model predicts:
- Failure risk score
- Risk level (Low / Medium / High)
- Maintenance recommendation

## Project structure
```
ai_predictive_maintenance/
├── api/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── style.css
│   └── requirements.txt
├── data/
│   └── generate_data.py
├── models/
│   ├── train_model.py
│   ├── predictive_model.joblib
│   └── feature_columns.json
├── docs/
│   └── sample_requests.json
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick start
### 1) Create data
```bash
python data/generate_data.py
```

### 2) Train the model
```bash
python models/train_model.py
```

### 3) Run the API
```bash
cd api
pip install -r requirements.txt
python app.py
```

Then open:
- Dashboard: http://127.0.0.1:5000
- Health check: http://127.0.0.1:5000/health

## Docker
```bash
docker compose up --build
```

## API endpoints
### GET /
Dashboard UI.

### GET /health
Returns service health and model load status.

### POST /predict
Example JSON body:
```json
{
  "temperature": 88,
  "vibration": 6.2,
  "pressure": 42,
  "rpm": 2950,
  "runtime_hours": 12400,
  "load_percent": 86,
  "oil_quality_index": 58
}
```

## Portfolio positioning
You can present this project as:
- AI-enabled predictive maintenance for industrial environments
- Flask-based real-time inference service
- Applied machine learning for equipment reliability
- Oil & gas / utilities digital transformation use case

## Suggested next upgrades
- Replace synthetic data with plant historian or SCADA exports
- Add time-series forecasting with LSTM
- Connect to PostgreSQL for telemetry history
- Add authentication and role-based access
- Add Grafana / Prometheus monitoring
