"""LightGBMによる学習・予測・提出ファイル出力をまとめるモジュール。

モデルの学習そのものと、推論結果の保存を一箇所で扱うことで使い方が明確になる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import lightgbm as lgb
except ModuleNotFoundError:  # pragma: no cover
    lgb = None

import numpy as np
import pandas as pd


def wmae_eval(preds: np.ndarray, train_data: Any) -> tuple[str, float, bool]:
    """LightGBMの評価関数として使うWMAE計算。

    学習ログに表示する指標と同じ定義で計算し、早期終了判定にも利用する。
    """
    y_true = train_data.get_label()
    weights = np.abs(y_true) * 1e4 + 1
    wmae = np.sum(weights * np.abs(y_true - preds)) / np.sum(weights)
    return "WMAE", float(wmae), False


def train_lightgbm_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    categorical_cols: list[str],
    model_params: dict[str, Any],
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 100,
):
    """LightGBMで回帰モデルを学習し、検証用データで評価する。"""
    if lgb is None:
        raise ModuleNotFoundError("lightgbm is required for training. Install it with: pip install lightgbm")

    # 学習用データセットと検証用データセットを作り、カテゴリ変数を認識させる。
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_cols)

    model = lgb.train(
        model_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        feval=wmae_eval,
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
    )
    return model


def predict_submission(model: Any, df_test: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """提出用テストデータに対して予測値を出す。"""
    prediction_df = df_test.reindex(columns=feature_list)
    return model.predict(prediction_df)


def save_submission(predictions: np.ndarray, output_path: str | Path = "submit.csv") -> pd.DataFrame:
    """予測結果をCSVとして保存する。

    1列目に予測値を出力し、提出用ファイルとして利用できるようにする。
    """
    output = pd.DataFrame(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=True, header=False)
    return output
