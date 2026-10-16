"""
KAIROS - Machine Learning Training & Evaluation CLI
Trains the NumPy Linear Regression model from scratch on chronological split
and evaluates against historical baselines.
Usage: python scripts/train_model.py
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
    print(" KAIROS - DEMAND FORECASTING ML TRAINING (FROM SCRATCH VIA NUMPY)")
    print(" Team: Laphing Buddha | DATATHON 2026")
    print("======================================================================\n")

    start_timer = time.time()
    raw_csv = r"C:\Users\DELL\Documents\CampusSpace-Intelligence\data\raw\room_requests.csv"
    if not os.path.exists(raw_csv):
        print(f"Error: {raw_csv} not found.")
        return

    reqs_df = pd.read_csv(raw_csv)
    print(f"Loaded {len(reqs_df):,} raw room requests from {raw_csv}")

    # 1. Feature Engineering & Time-Series Preparation
    ts_df = DemandForecaster.prepare_demand_series(reqs_df)
    X, y, feature_names = DemandForecaster.build_feature_matrix(ts_df)
    n_samples = len(X)
    print(f"Prepared {n_samples:,} hourly demand observations across {len(feature_names)} features:")
    for f in feature_names:
        print(f"  • {f}")

    # 2. Chronological Split (Strictly temporal: 70% Train, 15% Validation, 15% Test)
    # Zero future leakage: train comes strictly before validation, validation strictly before test
    train_end = int(0.70 * n_samples)
    val_end = int(0.85 * n_samples)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\nChronological Split:")
    print(f"  Train samples:      {len(X_train):,} ({ts_df['request_date'].iloc[0]} to {ts_df['request_date'].iloc[train_end-1]})")
    print(f"  Validation samples: {len(X_val):,} ({ts_df['request_date'].iloc[train_end]} to {ts_df['request_date'].iloc[val_end-1]})")
    print(f"  Test samples:       {len(X_test):,} ({ts_df['request_date'].iloc[val_end]} to {ts_df['request_date'].iloc[-1]})")

    # 3. Compute Historical Baselines on Test Set
    # Baseline 1: Overall historical training mean
    mean_val = np.mean(y_train)
    y_pred_baseline_mean = np.full_like(y_test, fill_value=mean_val)
    base_mean_metrics = calculate_metrics(y_test, y_pred_baseline_mean)

    # Baseline 2: Same-Weekday-Hour Average Baseline
    train_df = ts_df.iloc[:train_end]
    lookup_table = train_df.groupby(["weekday_num", "start_hour"])["request_count"].mean().to_dict()
    test_df = ts_df.iloc[val_end:]
    y_pred_baseline_stratified = np.array([
        lookup_table.get((w, h), mean_val)
        for w, h in zip(test_df["weekday_num"], test_df["start_hour"])
    ])
    base_strat_metrics = calculate_metrics(y_test, y_pred_baseline_stratified)

    print("\n----------------------------------------------------------------------")
    print("[BASELINE PERFORMANCE ON TEST SET]")
    print(f"  1. Overall Mean Baseline:           MAE = {base_mean_metrics['mae']:.4f} | RMSE = {base_mean_metrics['rmse']:.4f}")
    print(f"  2. Same-Weekday-Hour Mean Baseline: MAE = {base_strat_metrics['mae']:.4f} | RMSE = {base_strat_metrics['rmse']:.4f}")

    # 4. Train Pure NumPy Model from Scratch
    print("\n----------------------------------------------------------------------")
    print("[TRAINING NUMPY LINEAR REGRESSION FROM SCRATCH]")
    print("  Optimizer: Batch Gradient Descent with L2 Penalty")
    print("  Learning Rate: 0.05 | Epochs: 2,000 | L2 Reg: 0.01")
    
    model = CustomLinearRegression(learning_rate=0.05, epochs=2000, l2_penalty=0.01, random_state=42)
    fit_res = model.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
    print(f"  Initial Train MSE: {model.train_losses[0]:.4f} -> Final Train MSE: {fit_res['final_train_mse']:.4f}")
    print(f"  Best Validation MSE: {fit_res['best_val_mse']:.4f}")

    # 5. Evaluate on Test Set
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    improvement_mae = round(((base_strat_metrics["mae"] - test_metrics["mae"]) / base_strat_metrics["mae"]) * 100, 2)
    improvement_rmse = round(((base_strat_metrics["rmse"] - test_metrics["rmse"]) / base_strat_metrics["rmse"]) * 100, 2)
    elapsed_time = round(time.time() - start_timer, 3)

    print("\n----------------------------------------------------------------------")
    print("[FINAL TEST SET EVALUATION]")
    print(f"  Model Test MAE:  {test_metrics['mae']:.4f} requests")
    print(f"  Model Test RMSE: {test_metrics['rmse']:.4f} requests")
    print(f"  MAE Improvement vs Stratified Baseline: {improvement_mae:+}%")
    print(f"  RMSE Improvement vs Stratified Baseline: {improvement_rmse:+}%")
    print(f"  Total Runtime: {elapsed_time} seconds")

    # 6. Save Model Artifact & Metrics
    model_dir = r"C:\Users\DELL\Documents\CampusSpace-Intelligence\data"
    model_path = os.path.join(model_dir, "demand_forecast_model.json")
    model.save(model_path)
    print(f"\nSaved model weights to: {model_path}")

    metrics_path = os.path.join(model_dir, "ml_evaluation_metrics.json")
    eval_summary = {
        "model_name": "NumPy Linear Regression (Trained from scratch)",
        "framework": "Pure NumPy (No sklearn)",
        "task": "Campus Hourly Request Demand Forecasting",
        "target": "request_count (Hourly requests submitted)",
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "features": feature_names,
        "feature_weights": dict(zip(feature_names, [round(w, 4) for w in model.weights])),
        "bias": round(model.bias, 4),
        "baselines": {
            "overall_mean": base_mean_metrics,
            "same_weekday_hour_mean": base_strat_metrics
        },
        "model_test_metrics": test_metrics,
        "improvement_pct_mae": improvement_mae,
        "improvement_pct_rmse": improvement_rmse,
        "training_time_seconds": elapsed_time
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    print("\n======================================================================")
    print(" MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY.")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
