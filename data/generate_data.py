import os
import numpy as np
import pandas as pd


OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'equipment_telemetry.csv')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    temperature = rng.normal(75, 12, n_samples).clip(35, 130)
    vibration = rng.normal(4.5, 1.8, n_samples).clip(0.2, 12)
    pressure = rng.normal(55, 14, n_samples).clip(10, 110)
    rpm = rng.normal(2800, 500, n_samples).clip(800, 4200)
    runtime_hours = rng.normal(9000, 5000, n_samples).clip(50, 30000)
    load_percent = rng.normal(68, 18, n_samples).clip(10, 120)
    oil_quality_index = rng.normal(72, 15, n_samples).clip(10, 100)

    # Failure tendency logic with controlled noise
    risk_signal = (
        0.05 * (temperature - 75)
        + 0.30 * (vibration - 4.5)
        + 0.02 * (rpm - 2800) / 100
        + 0.00008 * (runtime_hours - 9000)
        + 0.03 * (load_percent - 68)
        - 0.06 * (oil_quality_index - 72)
        - 0.02 * (pressure - 55)
    )

    risk_signal += rng.normal(0, 0.9, n_samples)
    failure_probability = sigmoid(risk_signal)
    failure_within_30_days = (failure_probability > 0.55).astype(int)

    df = pd.DataFrame({
        'temperature': temperature.round(2),
        'vibration': vibration.round(2),
        'pressure': pressure.round(2),
        'rpm': rpm.round(0).astype(int),
        'runtime_hours': runtime_hours.round(0).astype(int),
        'load_percent': load_percent.round(2),
        'oil_quality_index': oil_quality_index.round(2),
        'failure_within_30_days': failure_within_30_days,
    })
    return df


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f'Dataset written to: {os.path.abspath(OUTPUT_PATH)}')
    print(dataset.head())
    print('\nClass balance:')
    print(dataset['failure_within_30_days'].value_counts(normalize=True).rename('ratio'))
