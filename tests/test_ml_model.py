"""
Tests for Custom NumPy Linear Regression and Demand Forecasting
"""

import pytest
import numpy as np
import pandas as pd
from src.ml_model import CustomLinearRegression, DemandForecaster


def test_custom_linear_regression_training_and_convergence():
    # Synthetic linear problem: y = 2*x1 + 3*x2 + 5
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 5.0 + np.random.normal(0, 0.05, 100)

    model = CustomLinearRegression(learning_rate=0.1, epochs=300, l2_penalty=0.0)
    fit_res = model.fit(X, y)

    assert fit_res["final_train_mse"] < 0.1
    y_pred = model.predict(X)
    mae = np.mean(np.abs(y - y_pred))
    assert mae < 0.25


def test_model_serialization(tmp_path):
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([5.0, 11.0, 17.0])
    model = CustomLinearRegression(epochs=100)
    model.fit(X, y)

    save_file = str(tmp_path / "test_model.json")
    model.save(save_file)

    loaded = CustomLinearRegression.load(save_file)
    assert np.allclose(model.weights, loaded.weights)
    assert np.isclose(model.bias, loaded.bias)
    assert np.allclose(model.predict(X), loaded.predict(X))


def test_demand_forecaster_feature_preparation():
    sample_reqs = pd.DataFrame([
        {"request_date": "2025-09-01", "start_hour": 10},
        {"request_date": "2025-09-01", "start_hour": 10},
        {"request_date": "2025-09-01", "start_hour": 14},
        {"request_date": "2025-09-02", "start_hour": 10}
    ])
    ts_df = DemandForecaster.prepare_demand_series(sample_reqs)
    X, y, f_names = DemandForecaster.build_feature_matrix(ts_df)
    assert len(X) == 3
    assert len(f_names) == 10
    assert X.shape[1] == 10
