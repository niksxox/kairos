"""
KAIROS - Machine Learning Demand Forecasting Module
Pure NumPy Implementation of Linear Regression Trained from Scratch.
No scikit-learn or blackbox ML libraries used for the model or training pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class CustomLinearRegression:
    """
    Linear Regression model implemented from scratch using NumPy.
    Supports gradient descent optimization, L2 regularization (Ridge),
    and model serialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 1500,
        l2_penalty: float = 0.01,
        random_state: int = 42
    ):
        self.lr = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_names: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Trains model weights using Batch Gradient Descent with L2 regularization.
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X_train.shape
        self.feature_names = feature_names or [f"f_{i}" for i in range(n_features)]

        # Fit feature standardizer (z-score scaling) on training set only (prevent leakage)
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0  # Avoid division by zero

        X_train_scaled = (X_train - self.scaler_mean) / self.scaler_std
        if X_val is not None:
            X_val_scaled = (X_val - self.scaler_mean) / self.scaler_std

        # Initialize weights with Xavier/Glorot scaling
        limit = np.sqrt(6.0 / (n_features + 1))
        self.weights = np.random.uniform(-limit, limit, size=(n_features,))
        self.bias = float(np.mean(y_train))

        self.train_losses = []
        self.val_losses = []

        best_val_loss = float("inf")
        best_weights = self.weights.copy()
        best_bias = self.bias

        # Gradient Descent Loop
        for epoch in range(self.epochs):
            # Forward pass: y_pred = X * w + b
            y_pred = np.dot(X_train_scaled, self.weights) + self.bias
            error = y_pred - y_train

            # Compute gradients with L2 regularization penalty
            dw = (1.0 / n_samples) * (np.dot(X_train_scaled.T, error) + self.l2_penalty * self.weights)
            db = (1.0 / n_samples) * np.sum(error)

            # Parameter update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Compute training MSE
            train_mse = float(np.mean(error ** 2))
            self.train_losses.append(train_mse)

            if X_val is not None and y_val is not None:
                val_pred = np.dot(X_val_scaled, self.weights) + self.bias
                val_mse = float(np.mean((val_pred - y_val) ** 2))
                self.val_losses.append(val_mse)

                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_weights = self.weights.copy()
                    best_bias = self.bias

        # Restore best parameters if validation was used
        if X_val is not None and y_val is not None:
            self.weights = best_weights
            self.bias = best_bias

        return {
            "final_train_mse": self.train_losses[-1],
            "best_val_mse": best_val_loss if X_val is not None else None,
            "weights": self.weights.tolist(),
            "bias": self.bias
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predictions for input matrix X. Clamps output to >= 0 for demand counts.
        """
        if self.weights is None or self.scaler_mean is None or self.scaler_std is None:
            raise ValueError("Model is not trained. Fit or load weights first.")

        X_scaled = (X - self.scaler_mean) / self.scaler_std
        raw_pred = np.dot(X_scaled, self.weights) + self.bias
        return np.maximum(0.0, raw_pred)

    def save(self, filepath: str):
        """Saves weights and normalization parameters to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "model_type": "CustomLinearRegression_FromScratch",
            "weights": self.weights.tolist() if self.weights is not None else [],
            "bias": self.bias,
            "feature_names": self.feature_names,
            "scaler_mean": self.scaler_mean.tolist() if self.scaler_mean is not None else [],
            "scaler_std": self.scaler_std.tolist() if self.scaler_std is not None else [],
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "l2_penalty": self.l2_penalty
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CustomLinearRegression":
        """Loads weights and parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        model = cls(
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            l2_penalty=payload["l2_penalty"]
        )
        model.weights = np.array(payload["weights"])
        model.bias = float(payload["bias"])
        model.feature_names = payload["feature_names"]
        model.scaler_mean = np.array(payload["scaler_mean"])
        model.scaler_std = np.array(payload["scaler_std"])
        return model


class DemandForecaster:
    """
    Feature engineering pipeline and end-to-end inference wrapper
    for campus hourly room request forecasting.
    """

    WEEKDAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model: Optional[CustomLinearRegression] = None
        if model_path and os.path.exists(model_path):
            self.model = CustomLinearRegression.load(model_path)

    @classmethod
    def prepare_demand_series(cls, room_requests_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates individual requests into an hourly time-series grid
        spanning all calendar dates and operating hours.
        """
        df = room_requests_df.copy()
        df["dt"] = pd.to_datetime(df["request_date"])
        
        # Aggregate count per date and hour
        grouped = df.groupby(["request_date", "start_hour"]).size().reset_index(name="request_count")
        grouped["dt"] = pd.to_datetime(grouped["request_date"])
        grouped["weekday"] = grouped["dt"].dt.strftime("%a")
        grouped["weekday_num"] = grouped["dt"].dt.weekday
        grouped["month"] = grouped["dt"].dt.month
        grouped["day"] = grouped["dt"].dt.day

        # Sort chronologically (critical for time-series validity and no leakage)
        grouped = grouped.sort_values(by=["dt", "start_hour"]).reset_index(drop=True)

        # Feature 1: Cyclic hour encoding
        h = grouped["start_hour"]
        grouped["sin_hour"] = np.sin(2 * np.pi * h / 24.0)
        grouped["cos_hour"] = np.cos(2 * np.pi * h / 24.0)

        # Feature 2: Cyclic weekday encoding
        w = grouped["weekday_num"]
        grouped["sin_weekday"] = np.sin(2 * np.pi * w / 7.0)
        grouped["cos_weekday"] = np.cos(2 * np.pi * w / 7.0)

        # Feature 3: Normalized linear hour
        grouped["norm_hour"] = h / 24.0

        # Feature 4: Rolling lag features (strictly backward-looking to prevent future leakage)
        # Shift 1 represents immediate previous recorded bucket
        grouped["lag_1"] = grouped["request_count"].shift(1).fillna(grouped["request_count"].mean())
        # Shift 13 represents approximate previous day same-time slot (13 slots/day)
        grouped["lag_day"] = grouped["request_count"].shift(13).fillna(grouped["request_count"].mean())
        # Rolling 7-step moving average of past demand
        grouped["rolling_mean_7"] = (
            grouped["request_count"]
            .shift(1)
            .rolling(window=7, min_periods=1)
            .mean()
            .fillna(grouped["request_count"].mean())
        )

        return grouped

    @classmethod
    def build_feature_matrix(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        feature_cols = [
            "norm_hour",
            "sin_hour",
            "cos_hour",
            "sin_weekday",
            "cos_weekday",
            "weekday_num",
            "month",
            "lag_1",
            "lag_day",
            "rolling_mean_7"
        ]
        X = df[feature_cols].values.astype(np.float64)
        y = df["request_count"].values.astype(np.float64)
        return X, y, feature_cols

    def predict_slot(
        self,
        date_str: str,
        start_hour: int,
        recent_history_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Generates demand prediction for a user-selected time slot.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        dt = pd.to_datetime(date_str)
        w_num = dt.weekday
        h = float(start_hour)

        # Default historical lag estimators if no live buffer passed
        avg_demand = 3.32
        lag_1 = avg_demand
        lag_day = avg_demand
        rolling_7 = avg_demand

        if recent_history_df is not None and not recent_history_df.empty:
            lag_1 = float(recent_history_df["request_count"].iloc[-1])
            if len(recent_history_df) >= 13:
                lag_day = float(recent_history_df["request_count"].iloc[-13])
            rolling_7 = float(recent_history_df["request_count"].tail(7).mean())

        x_vec = np.array([[
            h / 24.0,
            np.sin(2 * np.pi * h / 24.0),
            np.cos(2 * np.pi * h / 24.0),
            np.sin(2 * np.pi * w_num / 7.0),
            np.cos(2 * np.pi * w_num / 7.0),
            float(w_num),
            float(dt.month),
            lag_1,
            lag_day,
            rolling_7
        ]], dtype=np.float64)

        pred = float(self.model.predict(x_vec)[0])
        return {
            "date": date_str,
            "hour": start_hour,
            "weekday": dt.strftime("%A"),
            "predicted_requests": round(pred, 1),
            "rounded_request_count": max(1, int(round(pred))),
            "historical_mean_baseline": round(avg_demand, 1),
            "demand_intensity": "High" if pred >= 4.5 else ("Moderate" if pred >= 2.5 else "Low")
        }
