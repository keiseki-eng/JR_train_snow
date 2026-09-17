from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "00.config"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def load_project_config(
    config_path: str | Path | None = None,
    path_config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_root = Path(config_path) if config_path is not None else CONFIG_DIR / "config.yaml"
    path_root = Path(path_config_path) if path_config_path is not None else CONFIG_DIR / "path.yaml"
    return load_yaml_config(config_root), load_yaml_config(path_root)


def build_feature_columns(config: dict[str, Any]) -> dict[str, list[str]]:
    feature_list = list(config.get("FEATURE", {}).get("FEATURE_LIST", []))
    categorical_cols = list(config.get("FEATURE", {}).get("CATEGORICAL_COLS", []))

    weather = config.get("WEATHER_FEATURE", {})
    locations = weather.get("locations", [])
    variables = weather.get("variables", [])
    hours = weather.get("hours", [])
    w_categories = config.get("WEATHER_CATEGORY", {}).get("WEATHER_CATEGORY_LIST", [])

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
