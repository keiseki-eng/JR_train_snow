"""データ整理や可視化に使う補助ユーティリティをまとめたモジュール。

通過時刻の補間、特徴量名の正規化、簡単な可視化など、分析前処理で便利な処理をここにまとめる。
"""

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import japanize_matplotlib
except ModuleNotFoundError:  # pragma: no cover
    japanize_matplotlib = None


def _is_pass_value(value) -> bool:
    """値が「通過」表記かどうかを判定する。"""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() == "通過"
    return False


def _parse_time_to_minutes(value):
    """HH:MM形式の時刻文字列を分単位に変換する。"""
    if pd.isna(value):
        return np.nan
    if not isinstance(value, str):
        return np.nan

    text = value.strip().replace("：", ":")
    if text in {"", "通過"}:
        return np.nan

    try:
        hour_str, minute_str = text.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except (TypeError, ValueError):
        return np.nan

    return hour * 60 + minute


def _minutes_to_time_str(total_minutes):
    """分数をH:MM形式に変換する。"""
    total_minutes = int(round(float(total_minutes)))
    total_minutes = total_minutes % (24 * 60)
    hour, minute = divmod(total_minutes, 60)
    return f"{hour}:{minute:02d}"


def fill_pass_time_cells(series: pd.Series) -> pd.Series:
    """連続した「通過」セルを、左右の時刻の中間値で埋める。

    例: 6:00, 通過, 通過, 7:00 -> 6:00, 6:20, 6:40, 7:00
    """
    values = list(series.astype(object))
    filled = values.copy()
    i = 0

    while i < len(filled):
        if not _is_pass_value(filled[i]):
            i += 1
            continue

        start = i
        while i < len(filled) and _is_pass_value(filled[i]):
            i += 1
        end = i

        left_index = start - 1
        right_index = end

        # 通過セルの左右にある実時刻を探して、補間の基準とする。
        while left_index >= 0 and _is_pass_value(filled[left_index]):
            left_index -= 1
        while right_index < len(filled) and _is_pass_value(filled[right_index]):
            right_index += 1

        if left_index < 0 or right_index >= len(filled):
            continue

        left_minutes = _parse_time_to_minutes(filled[left_index])
        right_minutes = _parse_time_to_minutes(filled[right_index])

        if np.isnan(left_minutes) or np.isnan(right_minutes):
            continue

        pass_count = end - start
        if pass_count <= 0:
            continue

        if right_minutes == left_minutes:
            fill_value = left_minutes
            for idx in range(start, end):
                filled[idx] = _minutes_to_time_str(fill_value)
            continue

        step = (right_minutes - left_minutes) / (pass_count + 1)
        for offset, idx in enumerate(range(start, end), start=1):
            interpolated_minutes = left_minutes + step * offset
            filled[idx] = _minutes_to_time_str(interpolated_minutes)

    return pd.Series(filled, index=series.index)


def fill_pass_time_cells_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameの各行に対して、通過セルの補間を適用する。"""
    if df.empty:
        return df.copy()
    return df.apply(fill_pass_time_cells, axis=1)


def _normalize_feature_key(value) -> str:
    """特徴量名の比較を安定させるために記号や空白を正規化する。"""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).replace(" ", "")
    text = re.sub(r"[^0-9A-Za-zぁ-んァ-ン一-龥_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _infer_station_columns(df: pd.DataFrame) -> list[str]:
    """時刻文字列を持つ列を駅の通過時刻列として推定する。"""
    station_columns = []
    for col in df.columns:
        if col in {"年月日", "列車番号", "停車駅名", "合計", "回送列車フラグ", "停車時刻"}:
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        is_time_like = values.map(lambda x: isinstance(x, str) and re.fullmatch(r"\d{1,2}:\d{2}", x.strip()) is not None).all()
        if is_time_like:
            station_columns.append(col)

    return station_columns


def build_pass_time_weather_features(
    df: pd.DataFrame,
    weather_variables: list[str] | None = None,
    station_columns: list[str] | None = None,
    suffix: str = "列車通過時",
) -> pd.DataFrame:
    """各駅の列車通過時刻に対応する気象条件を特徴量として作る。

    例: 「金沢の列車通過時刻」に対応する気温を列として追加する。
    """
    if df.empty:
        return df.copy()

    output = df.copy()
    if weather_variables is None:
        weather_variables = []
    if station_columns is None:
        station_columns = _infer_station_columns(output)

    if not weather_variables:
        return output

    for station in station_columns:
        if station not in output.columns:
            continue

        station_norm = _normalize_feature_key(station)
        for variable in weather_variables:
            new_col = f"{station}_{suffix}_{variable}"
            values = []

            for _, row in output.iterrows():
                time_value = row.get(station)
                minutes = _parse_time_to_minutes(time_value)
                if pd.isna(minutes):
                    values.append(np.nan)
                    continue

                hour = int(minutes // 60) % 24
                hour_texts = [f"{hour}_00", f"{hour}:00", f"{hour}"]
                variable_norm = _normalize_feature_key(variable)
                matched_col = None

                for col in output.columns:
                    col_norm = _normalize_feature_key(col)
                    if not station_norm or station_norm not in col_norm:
                        continue
                    if not variable_norm or variable_norm not in col_norm:
                        continue
                    if not any(text in col_norm for text in hour_texts):
                        continue
                    matched_col = col
                    break

                if matched_col is None:
                    values.append(np.nan)
                    continue

                values.append(row.get(matched_col, np.nan))

            output[new_col] = values

    return output


def read_data(path: Path, **kwargs):
    """拡張子に応じてCSV/Excel/JSON/pickleを読み込む汎用関数。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("dtype", "str")
        return pd.read_csv(path, **kwargs)
    elif suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path, **kwargs)
    elif suffix == ".json":
        return pd.read_json(path, **kwargs)
    elif suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path, **kwargs)
    else:
        raise ValueError(f"対応していないファイル形式です: {suffix}")


def normalize_text(s: pd.Series) -> pd.Series:
    """文字列データの表記ゆれを統一する。

    全角/半角の差や空白、ハイフンの違いを揃えることで、列名や駅名の比較を安定させる。
    """
    normalized = s.astype("string").map(lambda x: unicodedata.normalize("NFKC", str(x)) if pd.notnull(x) else x)
    return (
        normalized.str.replace(r"\s+", "", regex=True)
        .str.replace(r"[-‐‑‒–—―]", "-", regex=True)
    )


def plot_feature_vs_target(
    df,
    feature_cols,
    target_col,
    bins=20,
    figsize=(8, 5),
):
    """指定した特徴量を横軸にして、目的変数との関係を可視化する。

    特徴量名の部分一致で対象列を選び、binごとの平均値と散布図を一緒に描く。
    """
    selected_features = []

    for feature in feature_cols:
        matched_cols = [col for col in df.columns if feature in col]
        selected_features.extend(matched_cols)

    selected_features = list(dict.fromkeys(selected_features))
    print(f"描画対象特徴量数: {len(selected_features)}")

    for feature in selected_features:
        x = pd.to_numeric(df[feature], errors="coerce")
        y = pd.to_numeric(df[target_col], errors="coerce")

        tmp = pd.DataFrame({"feature": x, "target": y}).dropna()

        if len(tmp) == 0:
            continue
        if tmp["feature"].nunique() <= 1:
            continue

        tmp["bin"] = pd.cut(tmp["feature"], bins=bins)
        summary = tmp.groupby("bin", observed=True)["target"].agg(["mean", "count"])
        x_mid = [interval.mid for interval in summary.index]

        plt.figure(figsize=figsize)
        plt.scatter(tmp["feature"], tmp["target"], alpha=0.25, s=15)
        plt.plot(x_mid, summary["mean"], marker="o", linewidth=2)
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f"{feature} vs {target_col}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()