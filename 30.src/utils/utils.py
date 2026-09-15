import re
import unicodedata
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import japanize_matplotlib
except ModuleNotFoundError:  # pragma: no cover
    japanize_matplotlib = None



def _is_pass_value(value) -> bool:
    """通過セルかどうかを判定する。"""
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
    """
    連続した「通過」セルを、左右の時刻の中間値で埋める。
    連続通過が複数ある場合は、端の時刻の間を通過セル数で等分する。
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
    """特徴量名の比較を安定させるための正規化。"""
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
    """
    各駅の列車通過時刻に対応する気象条件を、同じ行の気象特徴量から抽出して列に追加する。

    例:
        金沢_列車通過時_気温
        富山_列車通過時_降水量
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
    """
    ファイルの拡張子に応じてpd.DataFrameで読み込む関数

    Args:
        path (Path): 読み込むファイルのパス
        **kwargs: pd.read_csvやpd.read_excelに渡す追加の引数

    Returns:
        pd.DataFrame: 読み込んだデータ
    """
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
    """
    データフレームのカラムの文字列を正規化する関数
    - 全角数字・英字・記号を半角に変換する
    - 半角・全角の空白を削除
    - ハイフンに相当する記号を半角のハイフンに統一する
    

    Args:
        s (pd.Series): 正規化する文字列のSeries

    Returns:
        pd.Series: 正規化された文字列のSeries
        
    """
    normalized = s.astype("string").map(lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else x)
    return (
        normalized.str.replace(r'\s+', '', regex=True)  # 空白を削除
        .str.replace(r'[-‐‑‒–—―]', '-', regex=True)  # ハイフンに相当する記号を半角のハイフンに統一
    )


def plot_feature_vs_target(
    df,
    feature_cols,
    target_col,
    bins=20,
    figsize=(8, 5),
):
    """
    指定した文字列を含む特徴量を対象として、
    特徴量と目的変数の関係を描画する。

    例：
        feature_cols = ["富山_気温"]

    の場合、

        富山_気温_11_00
        富山_気温_12_00
        富山_気温_13_00
        ...

    のように「富山_気温」を含むカラムをすべて描画対象とする。

    Parameters
    ----------
    df : pandas.DataFrame
        分析対象データ
    feature_cols : list
        検索する特徴量名（部分一致）
    target_col : str
        目的変数の列名
    bins : int
        特徴量を分割するbin数
    figsize : tuple
        グラフサイズ
    """

    # --------------------------------
    # 描画対象のカラムを検索
    # --------------------------------
    selected_features = []

    for feature in feature_cols:

        matched_cols = [
            col for col in df.columns
            if feature in col
        ]

        selected_features.extend(matched_cols)

    # 重複削除
    selected_features = list(dict.fromkeys(selected_features))

    print(f"描画対象特徴量数: {len(selected_features)}")

    # --------------------------------
    # 各特徴量を描画
    # --------------------------------
    for feature in selected_features:

        # 数値型に変換
        x = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        y = pd.to_numeric(
            df[target_col],
            errors="coerce"
        )

        tmp = pd.DataFrame({
            "feature": x,
            "target": y
        }).dropna()

        if len(tmp) == 0:
            continue

        # 特徴量に値の種類がほとんどない場合はスキップ
        if tmp["feature"].nunique() <= 1:
            continue

        # --------------------------------
        # bin分割
        # --------------------------------
        tmp["bin"] = pd.cut(
            tmp["feature"],
            bins=bins
        )

        # --------------------------------
        # binごとの目的変数平均
        # --------------------------------
        summary = tmp.groupby(
            "bin",
            observed=True
        )["target"].agg(
            ["mean", "count"]
        )

        # binの中央値をx座標にする
        x_mid = [
            interval.mid
            for interval in summary.index
        ]

        # --------------------------------
        # 描画
        # --------------------------------
        plt.figure(figsize=figsize)

        # 各データ
        plt.scatter(
            tmp["feature"],
            tmp["target"],
            alpha=0.25,
            s=15
        )

        # binごとの平均値
        plt.plot(
            x_mid,
            summary["mean"],
            marker="o",
            linewidth=2
        )

        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.title(f"{feature} vs {target_col}")

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()