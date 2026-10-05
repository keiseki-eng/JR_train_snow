"""モデルの重要特徴量とSHAP可視化を保存するモジュール。

学習後にどの特徴量が効いているか、また説明変数の寄与を視覚的に確認できるように
するための補助処理。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some environments
    shap = None


def save_feature_importance(model: object, feature_names: list[str], output_path: str | Path) -> pd.DataFrame:
    """LightGBMの特徴量重要度をCSVとして保存する。"""
    if hasattr(model, "feature_importance_"):
        importance = model.feature_importance_
    elif hasattr(model, "feature_importance"):
        importance = model.feature_importance()
    else:
        raise AttributeError("model does not expose feature_importance_ or feature_importance()")

    importance = list(importance)

    if len(importance) != len(feature_names):
        raise ValueError(
            f"feature importance length ({len(importance)}) != feature count ({len(feature_names)})"
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def _normalize_shap_values(shap_values: object, feature_count: int) -> np.ndarray:
    """SHAP値を2次元配列に正規化する。"""
    if shap_values is None:
        raise ValueError("SHAP values are empty")

    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            values = shap_values[1]
        else:
            values = shap_values[0]
    else:
        values = shap_values

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, feature_count)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[-1] == 2:
        return values[:, :, 1]
    if values.ndim == 3:
        return values[:, :, 0]
    raise ValueError(f"Unsupported SHAP tensor shape: {values.shape}")


def save_shap_summary_plot(
    model: object,
    X: pd.DataFrame,
    output_dir: str | Path,
    *,
    max_display: int = 20,
) -> list[Path]:
    """SHAP要約図とウォーターフォール図を画像として保存する。"""
    if shap is None:
        raise ModuleNotFoundError("SHAP is not installed; cannot save SHAP plots.")
    if X.empty:
        raise ValueError("X must not be empty to compute SHAP values.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X)
    values = _normalize_shap_values(raw_values, X.shape[1])

    summary_path = output / "shap_summary.png"
    plt.figure(figsize=(12, 8))
    shap.summary_plot(values, X, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    sample = X.iloc[[0]].copy()
    sample_values = _normalize_shap_values(explainer.shap_values(sample), X.shape[1])
    explanation = shap.Explanation(
        values=sample_values[0],
        base_values=np.mean(sample_values[0]),
        data=sample.iloc[0].to_numpy(),
        feature_names=list(X.columns),
    )
    waterfall_path = output / "shap_waterfall_0.png"
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(explanation, max_display=max_display)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()

    return [summary_path, waterfall_path]
