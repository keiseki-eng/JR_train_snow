import importlib.util
from pathlib import Path

import pandas as pd

module_path = Path(__file__).resolve().parents[1] / "30.src" / "utils" / "utils.py"
spec = importlib.util.spec_from_file_location("jr_utils", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fill_pass_time_cells = module.fill_pass_time_cells
build_pass_time_weather_features = module.build_pass_time_weather_features


def test_single_pass_is_midpoint():
    s = pd.Series(["6:00", "通過", "6:30"])
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:15", "6:30"])
    pd.testing.assert_series_equal(result, expected)


def test_multiple_passes_are_evenly_split():
    s = pd.Series(["6:00", "通過", "通過", "7:00"])
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:20", "6:40", "7:00"])
    pd.testing.assert_series_equal(result, expected)


def test_non_pass_values_are_preserved():
    s = pd.Series(["6:00", "通過", "7:00", "7:20"]) 
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:30", "7:00", "7:20"])
    pd.testing.assert_series_equal(result, expected)


def test_station_pass_weather_features_use_pass_hour():
    df = pd.DataFrame(
        {
            "年月日": ["2024-01-01", "2024-01-01"],
            "金沢": ["6:10", "7:40"],
            "富山": ["8:15", "9:00"],
            "金沢_気温(℃)_6:00": [10.0, 12.0],
            "金沢_気温(℃)_7:00": [11.0, 13.0],
            "富山_気温(℃)_8:00": [20.0, 21.0],
            "富山_気温(℃)_9:00": [21.0, 22.0],
        }
    )

    result = build_pass_time_weather_features(df, weather_variables=["気温"], station_columns=["金沢", "富山"])

    pd.testing.assert_series_equal(result["金沢_列車通過時_気温"], pd.Series([10.0, 13.0], name="金沢_列車通過時_気温"))
    pd.testing.assert_series_equal(result["富山_列車通過時_気温"], pd.Series([20.0, 22.0], name="富山_列車通過時_気温"))
