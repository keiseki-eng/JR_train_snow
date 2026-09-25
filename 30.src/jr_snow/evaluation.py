"""評価指標と分布要約の計算を担当するモジュール。

モデルの性能を見るときに使う WMAE だけでなく、予測値や正解値の全体感も出す。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def _prepare_binary_target_and_score(y_true: Any, y_score: Any, positive_label: Any = 1) -> tuple[np.ndarray, np.ndarray]:
    """2値ラベルとスコアへ変換する。"""
    y_true_array = np.asarray(y_true).reshape(-1)
    y_score_array = np.asarray(y_score, dtype=float).reshape(-1)

    if y_true_array.size != y_score_array.size:
        raise ValueError("y_true and y_score must have the same length")

    unique_labels = np.unique(y_true_array)
    if unique_labels.size != 2:
        raise ValueError("ROC-AUC requires a binary target with exactly two classes")

    positive_value = positive_label
    if positive_value not in unique_labels:
        positive_value = unique_labels[-1]

    y_true_binary = (y_true_array == positive_value).astype(int)
    return y_true_binary, y_score_array


def compute_roc_auc(y_true: Any, y_score: Any, positive_label: Any = 1) -> float:
    """AUC (Area Under the ROC Curve) を計算する。"""
    y_true_binary, y_score_array = _prepare_binary_target_and_score(y_true, y_score, positive_label=positive_label)

    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true_binary, y_score_array))
    except ModuleNotFoundError:  # pragma: no cover
        pass

    positive_count = int(y_true_binary.sum())
    negative_count = int(len(y_true_binary) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC-AUC is undefined when only one class is present")

    sorted_indices = np.argsort(y_score_array)[::-1]
    sorted_scores = y_score_array[sorted_indices]
    sorted_labels = y_true_binary[sorted_indices]

    thresholds = np.unique(sorted_scores)
    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0

    for score in thresholds:
        mask = sorted_scores == score
        tp += int(sorted_labels[mask].sum())
        fp += int((1 - sorted_labels[mask]).sum())
        fpr.append(fp / negative_count)
        tpr.append(tp / positive_count)

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    auc = float(np.trapz(tpr, fpr))
    return auc


def compute_roc_curve(y_true: Any, y_score: Any, positive_label: Any = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC曲線のFPR, TPR, thresholdを返す。"""
    y_true_binary, y_score_array = _prepare_binary_target_and_score(y_true, y_score, positive_label=positive_label)

    try:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score_array)
        return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)
    except ModuleNotFoundError:  # pragma: no cover
        pass

    positive_count = int(y_true_binary.sum())
    negative_count = int(len(y_true_binary) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("ROC curve is undefined when only one class is present")

    sorted_indices = np.argsort(y_score_array)[::-1]
    sorted_scores = y_score_array[sorted_indices]
    sorted_labels = y_true_binary[sorted_indices]

    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0

    for score in np.unique(sorted_scores):
        mask = sorted_scores == score
        tp += int(sorted_labels[mask].sum())
        fp += int((1 - sorted_labels[mask]).sum())
        fpr.append(fp / negative_count)
        tpr.append(tp / positive_count)

    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    thresholds = np.asarray(np.unique(sorted_scores), dtype=float)
    return fpr, tpr, thresholds


def plot_roc_curve(
    y_true: Any,
    y_score: Any,
    ax: Any | None = None,
    title: str = "ROC Curve",
    label: str | None = None,
    positive_label: Any = 1,
):
    """ROC曲線を描画し、AUCと合わせて返す。"""
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for plotting ROC curves")

    auc = compute_roc_auc(y_true, y_score, positive_label=positive_label)
    fpr, tpr, _ = compute_roc_curve(y_true, y_score, positive_label=positive_label)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, label=f"{label or 'Model'} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return ax


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
