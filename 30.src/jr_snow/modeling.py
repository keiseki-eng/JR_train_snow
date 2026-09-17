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
    if lgb is None:
        raise ModuleNotFoundError("lightgbm is required for training. Install it with: pip install lightgbm")

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
    prediction_df = df_test.reindex(columns=feature_list)
    return model.predict(prediction_df)


def save_submission(predictions: np.ndarray, output_path: str | Path = "submit.csv") -> pd.DataFrame:
    output = pd.DataFrame(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=True, header=False)
    return output
