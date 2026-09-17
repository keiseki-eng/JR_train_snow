from __future__ import annotations

from typing import Any

import numpy as np


def compute_wmae(y_true: Any, y_pred: Any) -> float:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    weights = np.abs(y_true_array) * 1e4 + 1
    return float(np.sum(weights * np.abs(y_true_array - y_pred_array)) / np.sum(weights))


def summarize_prediction_stats(y_pred: Any) -> dict[str, float]:
    array = np.asarray(y_pred, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize_target_stats(y_true: Any) -> dict[str, float]:
    array = np.asarray(y_true, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }
