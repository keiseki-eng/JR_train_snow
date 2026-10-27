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

try:
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError:  # pragma: no cover
    LinearRegression = None

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
    """LightGBMで回帰モデルを学習し、検証用データで評価する。

    環境に LightGBM が入っていない場合は同期的な scikit-learn の線形回帰へ
    自動フォールバックし、CI やローカルの最小構成でも検証ループが動くようにする。
    """
    if lgb is None:
        if LinearRegression is None:
            raise ModuleNotFoundError("lightgbm is required for training. Install it with: pip install lightgbm")

        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

        # 学習用・検証用の列構成を統一する
    X_train = X_train.copy()
    X_valid = X_valid.reindex(columns=X_train.columns).copy()

    # 設定上のカテゴリ列に加え、object/category型の列も検出する
    category_columns = set(
        column for column in categorical_cols
        if column in X_train.columns
    )

    for frame in (X_train, X_valid):
        category_columns.update(
            frame.select_dtypes(include=["object", "category"]).columns
        )

    category_columns = sorted(category_columns)

    # 学習用・検証用でカテゴリの候補と順序を完全に統一する
    for column in category_columns:
        all_values = pd.concat(
            [X_train[column], X_valid[column]],
            ignore_index=True,
        ).dropna().astype(str)

        categories = sorted(all_values.unique().tolist())
        dtype = pd.CategoricalDtype(
            categories=categories,
            ordered=False,
        )

        X_train[column] = X_train[column].astype("string").astype(dtype)
        X_valid[column] = X_valid[column].astype("string").astype(dtype)

    # categorical_feature の指定は学習用データセットだけに行う。
    # 検証用は reference=lgb_train から設定を引き継ぐ。
    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=category_columns,
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=lgb_train,
        free_raw_data=False,
    )
    model = lgb.train(
        model_params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        feval=wmae_eval,
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
    )
    # Persist the category column names in DataFrame column order. LightGBM stores
    # category value lists, but not their column names; inference must restore both.
    model._jr_snow_categorical_cols = X_train.select_dtypes(
        include=["category"]
    ).columns.tolist()
    return model


def train_lightgbm_model_on_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    model_params: dict[str, Any],
    num_boost_round: int = 1000,
):
    """CV検証データを含む全学習データで最終モデルを学習する。

    最終学習では検証用データを切り出さず、学習可能な全データを使って最終モデルを
    1回だけ学習する。モデル選定は CV によって行うが、最終出力モデルはこの学習で
    作成したモデルを用いる。
    """
    if lgb is None:
        if LinearRegression is None:
            raise ModuleNotFoundError("lightgbm is required for training. Install it with: pip install lightgbm")

        model = LinearRegression()
        model.fit(X, y)
        return model

    X = X.copy()
    # The categorical column list must describe the actual pandas dtypes passed to
    # LightGBM, not just the configured names.
    category_columns = X.select_dtypes(include=["category"]).columns.tolist()
    dataset = lgb.Dataset(X, y, categorical_feature=category_columns)
    model = lgb.train(model_params, dataset, num_boost_round=num_boost_round)
    # Keep names in the same order as model.pandas_categorical.
    model._jr_snow_categorical_cols = category_columns
    return model


def align_prediction_features(model: Any, df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """学習済みモデルの列順・カテゴリ定義に推論用DataFrameを合わせる。

    LightGBMは学習時のpandasカテゴリ列と推論時のカテゴリ列が一致しないと
    ValueErrorを送出する。テストだけでなく検証データにも同じ処理を適用する。
    """
    model_features = (
        list(model.feature_name())
        if callable(getattr(model, "feature_name", None))
        else list(feature_list)
    )
    prediction_df = df.reindex(columns=model_features).copy()

    categorical_cols = getattr(model, "_jr_snow_categorical_cols", None)
    pandas_categorical = getattr(model, "pandas_categorical", None)

    if categorical_cols is not None and pandas_categorical is not None:
        if len(categorical_cols) != len(pandas_categorical):
            raise ValueError(
                "学習モデル内のカテゴリ列名とカテゴリ定義の数が一致しません: "
                f"columns={len(categorical_cols)}, categories={len(pandas_categorical)}, "
                f"categorical_cols={categorical_cols}"
            )

        categorical_set = set(categorical_cols)
        for column, categories in zip(categorical_cols, pandas_categorical):
            if column not in prediction_df.columns:
                raise ValueError(f"学習時のカテゴリ特徴量が推論データにありません: {column}")
            dtype = pd.CategoricalDtype(categories=list(categories), ordered=False)
            prediction_df[column] = prediction_df[column].astype("string").astype(dtype)

        # 学習時にカテゴリ型でなかった列を、入力データ側のdtypeに依存させない。
        for column in prediction_df.columns:
            if column not in categorical_set:
                prediction_df[column] = pd.to_numeric(prediction_df[column], errors="coerce")

    return prediction_df


def predict_submission(model: Any, df_test: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """学習時の特徴量順・カテゴリ定義を復元して提出用データを予測する。"""
    prediction_df = align_prediction_features(model, df_test, feature_list)
    return model.predict(prediction_df)

def apply_prediction_correction(predictions: np.ndarray, correction_amount: float) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    mask = predictions > 0
    predictions[mask] += correction_amount
    return predictions


def save_submission(predictions: np.ndarray, output_path: str | Path = "submit.csv") -> pd.DataFrame:
    """予測結果をCSVとして保存する。

    サンプル提出ファイルと同じ形式で、行番号 + 予測値の2列を保存する。
    """
    values = np.asarray(predictions, dtype=float).reshape(-1)
    output = pd.DataFrame({
        "row_id": np.arange(len(values), dtype=int),
        "予測値": values,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, header=False)
    return output
