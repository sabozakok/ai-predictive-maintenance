import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'equipment_telemetry.csv'
MODEL_PATH = BASE_DIR / 'predictive_model.joblib'
FEATURES_PATH = BASE_DIR / 'feature_columns.json'

FEATURE_COLUMNS = [
    'temperature',
    'vibration',
    'pressure',
    'rpm',
    'runtime_hours',
    'load_percent',
    'oil_quality_index',
]
TARGET_COLUMN = 'failure_within_30_days'


def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run data/generate_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)

    joblib.dump(pipeline, MODEL_PATH)
    with open(FEATURES_PATH, 'w', encoding='utf-8') as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Features saved to: {FEATURES_PATH}')
    print(f'ROC-AUC: {auc:.4f}')
    print('\nClassification report:\n')
    print(report)


if __name__ == '__main__':
    train()
