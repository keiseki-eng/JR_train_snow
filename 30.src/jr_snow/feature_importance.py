"""モデルの重要特徴量をCSVとして出力するモジュール。

学習後にどの特徴量が効いているかを確認できるようにするための補助処理。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
