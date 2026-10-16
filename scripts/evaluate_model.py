"""
KAIROS - Machine Learning Evaluation CLI
Evaluates the pre-trained pure NumPy Linear Regression model on held-out test data
against naive and stratified baselines.
Usage: python scripts/evaluate_model.py
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml_model import CustomLinearRegression, DemandForecaster


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    return {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4)}


def main():
    print("======================================================================")
    print(" KAIROS - DEMAND FORECASTING ML EVALUATION CLI")
    print(" Evaluates NumPy Linear Regression against Historical Baselines")
    print(" Team: Laphing Buddha | DATATHON 2026")
    print("======================================================================\n")

    model_path = Path("data") / "demand_forecast_model.json"
    metrics_path = Path("data") / "ml_evaluation_metrics.json"
    raw_csv = Path("data") / "raw" / "room_requests.csv"

    if not model_path.exists():
        print(f"Error: Trained model weights not found at {model_path}. Run scripts/train_model.py first.")
        sys.exit(1)

    if not raw_csv.exists():
        print(f"Error: Requests dataset not found at {raw_csv}.")
        sys.exit(1)

    # 1. Load Model
    model = CustomLinearRegression.load(str(model_path))
    print(f"Loaded trained NumPy model weights from: {model_path}")
    print(f"  Features ({len(model.feature_names)}): {', '.join(model.feature_names)}")
    print(f"  Bias term: {model.bias:.4f}")

    # 2. Prepare Data & Chronological Split
    reqs_df = pd.read_csv(raw_csv)
    ts_df = DemandForecaster.prepare_demand_series(reqs_df)
    X, y, feature_names = DemandForecaster.build_feature_matrix(ts_df)
    n_samples = len(X)

    train_end = int(0.70 * n_samples)
    val_end = int(0.85 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[val_end:], y[val_end:]
    test_df = ts_df.iloc[val_end:]

    print(f"\nChronological Test Split:")
    print(f"  Total hourly samples: {n_samples:,}")
    print(f"  Held-out test samples: {len(X_test):,} ({test_df['request_date'].iloc[0]} to {test_df['request_date'].iloc[-1]})")

    # 3. Baselines
    train_mean = np.mean(y_train)
    y_pred_base_mean = np.full_like(y_test, fill_value=train_mean)
    base_mean_metrics = calculate_metrics(y_test, y_pred_base_mean)

    train_df = ts_df.iloc[:train_end]
    lookup_table = train_df.groupby(["weekday_num", "start_hour"])["request_count"].mean().to_dict()
    y_pred_base_strat = np.array([
        lookup_table.get((w, h), train_mean)
        for w, h in zip(test_df["weekday_num"], test_df["start_hour"])
    ])
    base_strat_metrics = calculate_metrics(y_test, y_pred_base_strat)

    # 4. Predict with NumPy Model
    y_pred_model = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_model)

    improvement_mae = round(((base_strat_metrics["mae"] - test_metrics["mae"]) / base_strat_metrics["mae"]) * 100, 2)
    improvement_rmse = round(((base_strat_metrics["rmse"] - test_metrics["rmse"]) / base_strat_metrics["rmse"]) * 100, 2)

    print("\n----------------------------------------------------------------------")
    print("EVALUATION RESULTS COMPARISON:")
    print(f"  1. Global Mean Baseline:           MAE = {base_mean_metrics['mae']:.4f} | RMSE = {base_mean_metrics['rmse']:.4f}")
    print(f"  2. Stratified Weekday-Hour Mean:   MAE = {base_strat_metrics['mae']:.4f} | RMSE = {base_strat_metrics['rmse']:.4f}")
    print(f"  3. KAIROS NumPy Linear Regression: MAE = {test_metrics['mae']:.4f} | RMSE = {test_metrics['rmse']:.4f}")
    print("----------------------------------------------------------------------")
    print(f"  MAE Improvement vs Stratified Baseline: {improvement_mae:+}%")
    print(f"  RMSE Improvement vs Stratified Baseline: {improvement_rmse:+}%")
    print("======================================================================\n")

    # Update or verify metrics file
    eval_summary = {
        "model_name": "NumPy Linear Regression (Trained from scratch)",
        "framework": "Pure NumPy (No sklearn)",
        "task": "Campus Hourly Request Demand Forecasting",
        "target": "request_count (Hourly requests submitted)",
        "train_samples": len(X_train),
        "val_samples": len(X) - len(X_train) - len(X_test),
        "test_samples": len(X_test),
        "features": feature_names,
        "feature_weights": dict(zip(feature_names, [round(float(w), 4) for w in model.weights])),
        "bias": round(float(model.bias), 4),
        "baselines": {
            "overall_mean": base_mean_metrics,
            "same_weekday_hour_mean": base_strat_metrics
        },
        "model_test_metrics": test_metrics,
        "improvement_pct_mae": improvement_mae,
        "improvement_pct_rmse": improvement_rmse,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"Evaluation report verified and updated at: {metrics_path}")


if __name__ == "__main__":
    main()
