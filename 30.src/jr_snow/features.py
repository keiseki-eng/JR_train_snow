from __future__ import annotations

from typing import Any

import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "年月日" not in output.columns:
        return output

    output["年月日"] = pd.to_datetime(output["年月日"])
    output["年"] = output["年月日"].dt.year
    output["月"] = output["年月日"].dt.month
    output["日"] = output["年月日"].dt.day
    output["曜日"] = output["年月日"].dt.dayofweek
    return output


def add_temperature_threshold_features(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    output = df.copy()
    for column in [column for column in output.columns if "気温" in column]:
        new_column = f"{column}_ge_{threshold}_C"
        output[new_column] = (pd.to_numeric(output[column], errors="coerce") >= threshold).astype("int8")
    return output


def build_engineered_feature_frame(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    output = add_date_features(df)
    output = add_temperature_threshold_features(output, threshold=threshold)
    return output


def split_train_valid(
    df_train: pd.DataFrame,
    split_date: str | pd.Timestamp,
    target_col: str = "合計",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    import sys
    from pathlib import Path

    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from utils.validation import split_time

    return split_time(df_train, split_col="年月日", target_col=target_col, split_date=pd.Timestamp(split_date))


def prepare_model_inputs(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: dict[str, Any],
    split_date: str,
    target_col: str = "合計",
    threshold: float = 5.0,
) -> dict[str, Any]:
    feature_list = list(feature_columns["feature_list"])
    categorical_cols = list(feature_columns["categorical_cols"])

    df_train = build_engineered_feature_frame(df_train, threshold=threshold)
    df_test = build_engineered_feature_frame(df_test, threshold=threshold)

    engineered_cols = [
        column for column in df_train.columns if column not in feature_list and "ge_5.0_C" in column
    ]
    feature_list = list(dict.fromkeys(feature_list + engineered_cols))

    fix_list = feature_list + ["年月日", target_col]
    df_train = df_train.reindex(columns=fix_list)

    X_train, X_valid, y_train, y_valid = split_train_valid(df_train, split_date=split_date, target_col=target_col)

    X_train = X_train[feature_list].copy()
    X_valid = X_valid[feature_list].copy()
    df_test_processed = df_test.reindex(columns=feature_list)

    cat_cols = [column for column in categorical_cols if column in X_train.columns]
    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    for frame in [X_train, X_valid, df_test_processed]:
        columns = sorted(set(cat_cols + obj_cols).intersection(frame.columns))
        if columns:
            frame[columns] = frame[columns].astype("category")

    for column in [column for column in X_train.columns if "日照時間" in column]:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce")
        df_test_processed[column] = pd.to_numeric(df_test_processed[column], errors="coerce")

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "df_test_processed": df_test_processed,
        "feature_list": feature_list,
        "categorical_cols": cat_cols,
    }
