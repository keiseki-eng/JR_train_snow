"""モデルの重要特徴量をCSVとして出力するモジュール。

学習後にどの特徴量が効いているかを確認できるようにするための補助処理。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_feature_importance(model: object, feature_names: list[str], output_path: str | Path) -> pd.DataFrame:
    """LightGBMの特徴量重要度をCSVとして保存する。

    重要度の大きい順に並べ替え、分析時に見やすい形式にする。
    """
    if not hasattr(model, "feature_importance_"):
        raise AttributeError("model does not expose feature_importance_")

    importance = model.feature_importance_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
