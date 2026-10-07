"""特徴量の生成と学習用データセットの組み立てを行うモジュール。

元データをそのまま使うのではなく、日付特徴量や閾値特徴量を加えて学習可能な形に整える。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from utils.utils import add_region_time_weather_features


def add_document_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """資料-1〜3 に基づく地域・時間帯ごとの降雪量・日射量・天気期待値を追加する。"""
    return add_region_time_weather_features(df)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """年月日を分解して、年・月・日・曜日を追加する。"""
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
    """気温が閾値以上かどうかを0/1の特徴量として追加する。"""
    output = df.copy()
    for column in [column for column in output.columns if "気温" in column]:
        new_column = f"{column}_ge_{threshold}_C"
        output[new_column] = (pd.to_numeric(output[column], errors="coerce") >= threshold).astype("int8")
    return output


def fill_missing_by_same_day_time_location_average(df: pd.DataFrame) -> pd.DataFrame:
    """同じ日・同じ時間帯で、他地点の平均値または最頻値を使って欠損値を補完する。

    数値特徴量は同一日・同一時刻の他地点平均で補完し、風向きのような文字列特徴量は
    同一定義の最頻値で補完する。最頻値が同率の場合は、最初に出現した値を優先する。
    """
    output = df.copy()
    if output.empty:
        return output

    def parse_location_metric_hour(column_name: str) -> tuple[str, str, str] | None:
        patterns = [
            r"^(?P<location>.+?)_(?P<metric>.+?)__(?P<hour>\d+_00)$",
            r"^(?P<location>.+?)_(?P<metric>.+)_(?P<hour>\d+_00)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, column_name)
            if match is None:
                continue
            return match.group("location"), match.group("metric"), match.group("hour")
        return None

    def most_frequent_first(values: pd.Series) -> object | float | None:
        non_null = values.dropna()
        if non_null.empty:
            return None
        counts: dict[object, int] = {}
        first_index: dict[object, int] = {}
        for index, value in enumerate(non_null.tolist()):
            if value not in counts:
                counts[value] = 0
                first_index[value] = index
            counts[value] += 1
        max_count = max(counts.values())
        candidates = [value for value, count in counts.items() if count == max_count]
        return min(candidates, key=lambda value: first_index[value])

    def column_is_numeric_like(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        if pd.api.types.is_numeric_dtype(non_null):
            return True
        converted = pd.to_numeric(non_null, errors="coerce")
        if converted.notna().sum() == 0:
            return False
        return converted.notna().sum() >= max(1, int(len(non_null) * 0.8))

    def ensure_numeric_column_if_possible(column: str, *, force: bool = False) -> None:
        if column not in output.columns:
            return
        if force or column_is_numeric_like(output[column]):
            output[column] = pd.to_numeric(output[column], errors="coerce")

    def assign_value_safely(column: str, index: Any, value: Any) -> None:
        if column not in output.columns:
            return
        if isinstance(output[column].dtype, pd.CategoricalDtype):
            output[column] = output[column].astype(object)
        output.at[index, column] = value

    # 1) 既存の long-format 形式（地点列があるケース）を優先的に処理する。
    if "地点" in output.columns:
        if "年月日時" in output.columns:
            output["_補完時刻"] = pd.to_datetime(output["年月日時"], errors="coerce")
        elif "年月日" in output.columns:
            output["_補完時刻"] = pd.to_datetime(output["年月日"], errors="coerce")
            if "時刻" in output.columns:
                hour_values = pd.to_numeric(output["時刻"], errors="coerce").fillna(0)
                output["_補完時刻"] = output["_補完時刻"] + pd.to_timedelta(hour_values.astype(int), unit="h")
        else:
            return output

        candidates: list[str] = []
        for column in output.columns:
            if column in {"年月日", "年月日時", "地点", "_補完時刻", "時刻"}:
                continue
            if output[column].isna().sum() == 0:
                continue
            candidates.append(column)

        for column in candidates:
            na_idx = output.index[output[column].isna()].tolist()
            for idx in na_idx:
                location = output.at[idx, "地点"]
                target_time = output.at[idx, "_補完時刻"]
                same_time_rows = output.loc[output["_補完時刻"] == target_time]
                if same_time_rows.empty:
                    continue

                other_locations = same_time_rows.loc[same_time_rows["地点"] != location, column]
                candidate_values = other_locations.dropna()
                if candidate_values.empty:
                    candidate_values = same_time_rows[column].dropna()
                if candidate_values.empty:
                    continue

                numeric_candidate_values = pd.to_numeric(candidate_values, errors="coerce")
                if numeric_candidate_values.notna().any():
                    ensure_numeric_column_if_possible(column, force=True)
                    assign_value_safely(column, idx, float(numeric_candidate_values.mean()))
                elif column_is_numeric_like(output[column]):
                    ensure_numeric_column_if_possible(column)
                    assign_value_safely(column, idx, float(pd.to_numeric(output[column], errors="coerce").mean()))
                else:
                    assign_value_safely(column, idx, most_frequent_first(candidate_values))

        return output.drop(columns=["_補完時刻"], errors="ignore")

    # 2) wide-format 形式（例: 富山_気温_℃__1_00）の補完。
    all_wide_columns = [
        column for column in output.columns
        if column not in {"年月日", "年月日時", "時刻", "停車時刻", "列車番号"}
        and parse_location_metric_hour(column) is not None
    ]

    for column in list(all_wide_columns):
        if output[column].notna().all():
            continue
        parsed = parse_location_metric_hour(column)
        if parsed is None:
            continue
        _, metric_name, hour_label = parsed
        other_location_columns = [
            candidate for candidate in all_wide_columns
            if candidate != column and parse_location_metric_hour(candidate) is not None
            and parse_location_metric_hour(candidate)[1] == metric_name
            and parse_location_metric_hour(candidate)[2] == hour_label
        ]
        if not other_location_columns:
            continue

        missing_mask = output[column].isna()
        if not missing_mask.any():
            continue
        for idx in output.index[missing_mask]:
            candidate_values = output.loc[idx, other_location_columns]
            candidate_values = candidate_values.dropna()
            if candidate_values.empty:
                continue
            numeric_candidate_values = pd.to_numeric(candidate_values, errors="coerce")
            if numeric_candidate_values.notna().any():
                ensure_numeric_column_if_possible(column, force=True)
                assign_value_safely(column, idx, float(numeric_candidate_values.mean()))
            elif column_is_numeric_like(output[column]):
                ensure_numeric_column_if_possible(column)
                assign_value_safely(column, idx, float(pd.to_numeric(output[column], errors="coerce").mean()))
            else:
                assign_value_safely(column, idx, most_frequent_first(candidate_values))

    # 3) 残った NaN は同日平均で埋め、最後に全体平均または全体最頻値で埋める。
    for column in [column for column in output.columns if column not in {"年月日", "年月日時", "時刻", "停車時刻"}]:
        if output[column].isna().sum() == 0:
            continue
        if "年月日" in output.columns:
            if column_is_numeric_like(output[column]):
                ensure_numeric_column_if_possible(column)
                date_fill = output.groupby("年月日")[column].transform("mean")
                output[column] = output[column].fillna(date_fill)
            else:
                date_fill = output.groupby("年月日")[column].transform(most_frequent_first)
                output[column] = output[column].fillna(date_fill)

        if output[column].isna().sum() == 0:
            continue
        if column_is_numeric_like(output[column]):
            ensure_numeric_column_if_possible(column)
            overall_fill = float(pd.to_numeric(output[column], errors="coerce").mean())
            output[column] = output[column].fillna(overall_fill)
        else:
            overall_fill = most_frequent_first(output[column])
            if overall_fill is not None:
                output[column] = output[column].fillna(overall_fill)

    return output


def build_engineered_feature_frame(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """日付特徴量を先に作成し、その後に欠損補完とその他の特徴量を追加する。

    これにより、補完の基準となる日付・時刻情報が先に揃ってから、より安定した
    地域・時間帯特徴量や閾値特徴量を生成できる。
    """
    output = add_date_features(df)
    output = fill_missing_by_same_day_time_location_average(output)
    output = add_document_weather_features(output)
    output = add_temperature_threshold_features(output, threshold=threshold)
    return output


def split_train_valid(
    df_train: pd.DataFrame,
    split_date: str | pd.Timestamp,
    target_col: str = "合計",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """指定した日付でtrain/validを分割する。

    時系列データのように未来を予測する問題では、学習時に未来情報を使わないようにする。
    """
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
    """学習・検証・提出用にデータを整形して返す。

    1. 追加特徴量を作る
    2. train/validを日付で分割する
    3. 型変換と数値変換を行う
    4. 最終的な特徴量リストを返す
    """
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

    # LightGBMはカテゴリ変数に対して明示的な型指定が必要なため、カテゴリ列を変換する。
    for frame in [X_train, X_valid, df_test_processed]:
        columns = sorted(set(cat_cols + obj_cols).intersection(frame.columns))
        if columns:
            frame[columns] = frame[columns].astype("category")

    # 日照時間の文字列が混ざっている場合は数値化してモデル入力に適した形式に整える。
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
