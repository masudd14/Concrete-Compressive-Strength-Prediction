"""Regression performance metrics and evaluation helpers."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive regression metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary of MAE, RMSE, R2, and Percentage Error relative to mean target.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mean_val = float(np.mean(y_true))
    error_pct = (rmse / mean_val) * 100 if mean_val != 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "mean_target": mean_val,
        "rmse_percentage_error": error_pct,
    }
