"""設定ファイルと特徴量定義を読み込むためのモジュール。

YAMLで管理されるモデル設定から、どの列を特徴量とするかを自動生成する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "00.config"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """YAMLファイルを読み込んで辞書に変換する。

    Args:
        path: 設定ファイルのPathまたは文字列

    Returns:
        dict[str, Any]: YAMLの内容をそのまま辞書にした値
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_project_config(
    config_path: str | Path | None = None,
    path_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """メイン設定とファイルパス設定をまとめて読み込む。

    これにより、学習条件とデータ配置を分離して管理しやすくする。
    """
    config_root = Path(config_path) if config_path is not None else CONFIG_DIR / "config.yaml"
    path_root = Path(path_config_path) if path_config_path is not None else CONFIG_DIR / "path.yaml"
    return load_yaml_config(config_root), load_yaml_config(path_root)


def build_feature_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    """設定ファイルから特定の命名規則で特徴量列を組み立てる。

    例: 気象データや雪データを、位置・変数・時間帯の組み合わせで自動生成する。
    """
    feature_list = list(config.get("FEATURE", {}).get("FEATURE_LIST", []))
    categorical_cols = list(config.get("FEATURE", {}).get("CATEGORICAL_COLS", []))

    weather = config.get("WEATHER_FEATURE", {})
    locations = weather.get("locations", [])
    variables = weather.get("variables", [])
    hours = weather.get("hours", [])
    w_categories = config.get("WEATHER_CATEGORY", {}).get("WEATHER_CATEGORY_LIST", [])

    # 例: 富山_気温_1_00, 金沢_降水量_12_00 のような特徴量を自動生成する。
    weather_feature_cols = [
        f"{location}_{variable}_{hour}"
        for location in locations
        for variable in variables
        for hour in hours
    ]
    weather_categorical_cols = [
        f"{location}_{category}_{hour}"
        for location in locations
        for category in w_categories
        for hour in hours
    ]

    snow = config.get("SNOW_FEATURE", {})
    snow_locations = snow.get("locations", [])
    snow_variables = snow.get("variables", [])
    snow_statistics = snow.get("statistics", [])
    snow_time_zones = snow.get("time_zones", [])
    snow_categories = config.get("SNOW_CATEGORY", {}).get("SNOW_CATEGORY_LIST", [])

    # 雪データでは位置・指標・統計量・時間帯の組み合わせで列名を作る。
    snow_feature_cols = [
        f"{location}_{variable}_{statistic}_{time_zone}"
        for location in snow_locations
        for variable in snow_variables
        for statistic in snow_statistics
        for time_zone in snow_time_zones
    ]
    snow_categorical_cols = [
        f"{location}_{snow_category}_{time_zone}"
        for location in snow_locations
        for snow_category in snow_categories
        for time_zone in snow_time_zones
    ]

    final_feature_list = feature_list + weather_feature_cols + snow_feature_cols
    final_categorical_cols = categorical_cols + weather_categorical_cols + snow_categorical_cols

    return {
        "base_features": feature_list,
        "weather_features": weather_feature_cols,
        "snow_features": snow_feature_cols,
        "feature_list": final_feature_list,
        "categorical_cols": final_categorical_cols,
    }
