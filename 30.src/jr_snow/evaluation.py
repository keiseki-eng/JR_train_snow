"""評価指標と分布要約の計算を担当するモジュール。

モデルの性能を見るときに使う WMAE だけでなく、予測値や正解値の全体感も出す。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_wmae(y_true: Any, y_pred: Any) -> float:
    """重み付き平均絶対誤差を計算する。

    実際の売り上げや雪量のように大きい値に対して誤差を相対的に重視するために用いる。
    """
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    weights = np.abs(y_true_array) * 1e4 + 1
    return float(np.sum(weights * np.abs(y_true_array - y_pred_array)) / np.sum(weights))


def summarize_prediction_stats(y_pred: Any) -> dict[str, float]:
    """予測値の最小・最大・平均を簡潔にまとめる。"""
    array = np.asarray(y_pred, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize_target_stats(y_true: Any) -> dict[str, float]:
    """正解値の分布を簡単に把握するための要約情報を返す。"""
    array = np.asarray(y_true, dtype=float)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }
