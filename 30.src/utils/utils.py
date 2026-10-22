"""データ整理や可視化に使う補助ユーティリティをまとめたモジュール。

通過時刻の補間、特徴量名の正規化、簡単な可視化など、分析前処理で便利な処理をここにまとめる。
"""

import re
import unicodedata
from pathlib import Path
from typing import Any

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


def fill_missing_by_same_day_time_location_average(df: pd.DataFrame) -> pd.DataFrame:
    """同じ日・同じ時間帯で、他地点の平均値または最頻値を使って欠損値を補完する。

    通過時刻の気温など、駅単位の時刻対応特徴量を生成する前にこの補完を行うことで、
    駅ごとの対応列が NaN になるケースを防ぐ。
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

        candidates = []
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

    重要: まず地点・時刻ごとの気象列に対して欠損補完を行い、その後に各駅の
    通過時刻に対応する気象値を作る。これにより、対応する時刻の気温が NaN
    のまま放置されることを避ける。
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

    output = fill_missing_by_same_day_time_location_average(output)

    def candidate_hour_value(column_name: str) -> int | None:
        normalized = _normalize_feature_key(column_name)
        match = re.search(r"(\d+)(?:_00|:00)?$", normalized)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    for station in station_columns:
        if station not in output.columns:
            continue

        station_norm = _normalize_feature_key(station)
        for variable in weather_variables:
            new_col = f"{station}_{suffix}_{variable}"
            values = []
            variable_norm = _normalize_feature_key(variable)
            candidate_cols = [
                col for col in output.columns
                if station_norm in _normalize_feature_key(col)
                and variable_norm in _normalize_feature_key(col)
            ]

            for _, row in output.iterrows():
                time_value = row.get(station)
                minutes = _parse_time_to_minutes(time_value)
                if pd.isna(minutes):
                    values.append(np.nan)
                    continue

                hour = int(minutes // 60) % 24
                matched_col = None
                exact_matches = [
                    col for col in candidate_cols
                    if candidate_hour_value(col) == hour
                ]
                if exact_matches:
                    matched_col = exact_matches[0]
                else:
                    nearest = min(
                        candidate_cols,
                        key=lambda col: abs(candidate_hour_value(col) - hour) if candidate_hour_value(col) is not None else np.inf,
                    )
                    matched_col = nearest if candidate_hour_value(nearest) is not None else None

                if matched_col is None:
                    values.append(np.nan)
                    continue

                value = row.get(matched_col, np.nan)
                if pd.isna(value):
                    other_values = [
                        pd.to_numeric(row.get(col), errors="coerce")
                        for col in candidate_cols
                        if pd.notna(row.get(col))
                    ]
                    if other_values:
                        value = float(np.mean(other_values))

                values.append(value)

            output[new_col] = values

    return output


def _pick_column(columns, *candidates: str) -> str | None:
    """候補列名の中から、存在する列を優先して返す。"""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def estimate_snowfall_amount(temperature_c, precipitation_mm):
    """資料-1 の経験式から降雪量を推定する。

    ρs = 53.6 exp(0.488 Ta) + 37.0
    X = ρs * Y / 10000
    ここで X は cm, Y は mm, Ta は 気温(℃) である。
    """
    temp = pd.to_numeric(temperature_c, errors="coerce")
    precip = pd.to_numeric(precipitation_mm, errors="coerce")
    snow_density = 53.6 * np.exp(0.488 * temp) + 37.0
    snowfall = (snow_density * precip) / 10000.0
    return snowfall.clip(lower=0.0)


def estimate_solar_radiation_from_sunshine(sunshine_hours):
    """資料-2 の日照率と地上日射量の関係から推定日射量を計算する。"""
    hours = pd.to_numeric(sunshine_hours, errors="coerce").fillna(0.0)
    ratio = hours.clip(lower=0.0, upper=1.0)
    base = np.where(ratio > 0.0, 0.244 + 0.511 * ratio, 0.118)
    return pd.Series(base, index=hours.index if hasattr(hours, "index") else None)


def estimate_expected_weather_value(weather_df: pd.DataFrame, value_col: str | None = None) -> pd.DataFrame:
    """地域 × 時刻ごとの天気の期待値を計算する。

    本来は天気種別と日照時間の分布を利用するが、ここでは同一地域・同一時間帯で
    観測した日照時間の平均値を期待値として扱う。
    """
    df = weather_df.copy()
    if df.empty:
        return df

    if "年月日時" in df.columns:
        df["時刻"] = pd.to_datetime(df["年月日時"]).dt.hour
    elif "時刻" not in df.columns:
        return df

    if value_col is None:
        value_col = _pick_column(df.columns, "日照時間( 時間)", "日照時間_時間_", "日照時間")
    if value_col is None:
        return df

    expected = (
        df.groupby(["地点", "時刻"], as_index=True)[value_col]
        .mean()
        .rename("天気期待値")
        .reset_index()
    )
    return expected


def add_region_time_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """地域・時間ごとの降雪量・日射量・天気期待値を、同時刻の気象データから作成する。"""
    output = df.copy()
    if output.empty:
        return output

    if "地点" not in output.columns:
        return output

    if "年月日時" in output.columns:
        output["時刻"] = pd.to_datetime(output["年月日時"]).dt.hour
    elif "時刻" not in output.columns:
        return output

    temp_col = _pick_column(output.columns, "気温(℃)", "気温")
    precip_col = _pick_column(output.columns, "降水量(mm)", "降水量")
    sunshine_col = _pick_column(output.columns, "日照時間( 時間)", "日照時間_時間_", "日照時間")
    weather_col = _pick_column(output.columns, "天気")
    if temp_col is None or precip_col is None or sunshine_col is None:
        return output

    output["__降雪量"] = estimate_snowfall_amount(output[temp_col], output[precip_col])
    output["__日射量"] = estimate_solar_radiation_from_sunshine(output[sunshine_col])

    expected = estimate_expected_weather_value(output, value_col=sunshine_col)
    expected_lookup = {
        (row["地点"], int(row["時刻"])): row["天気期待値"]
        for _, row in expected.iterrows()
    }

    for region in sorted(output["地点"].dropna().unique()):
        for hour in sorted(output["時刻"].dropna().astype(int).unique()):
            hour_label = f"{int(hour)}_00"
            snowfall = output.loc[(output["地点"] == region) & (output["時刻"] == hour), "__降雪量"].mean()
            radiation = output.loc[(output["地点"] == region) & (output["時刻"] == hour), "__日射量"].mean()
            expectation = expected_lookup.get((region, int(hour)), np.nan)
            output[f"{region}_降雪量_{hour_label}"] = np.where(
                (output["地点"] == region) & (output["時刻"] == hour), snowfall, np.nan
            )
            output[f"{region}_日射量_{hour_label}"] = np.where(
                (output["地点"] == region) & (output["時刻"] == hour), radiation, np.nan
            )
            output[f"{region}_天気期待値_{hour_label}"] = np.where(
                (output["地点"] == region) & (output["時刻"] == hour), expectation, np.nan
            )

    output = output.drop(columns=["__降雪量", "__日射量"], errors="ignore")
    return output


def read_data(path: Path, **kwargs):
    """拡張子に応じてCSV/Excel/JSON/pickleを読み込む汎用関数。

    日本語を含む CSV は UTF-8 ではなく CP932/Shift_JIS で保存されることが多く、
    そのまま UTF-8 と仮定すると文字化けする。ここでは候補を順に試して最初に
    成功したエンコーディングで読み込むようにする。
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        encoding = kwargs.pop("encoding", None)
        candidates = [encoding] if encoding else ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp"]
        last_error = None

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                kwargs.setdefault("dtype", "str")
                return pd.read_csv(path, encoding=candidate, **kwargs)
            except (UnicodeDecodeError, UnicodeError, ValueError) as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
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