"""通過時刻の補間と気象特徴量生成のテスト。

駅の通過時刻に対して、補間や対応する気象条件の抽出が適切に動くかを確認する。
"""

import importlib.util
from pathlib import Path

import pandas as pd

module_path = Path(__file__).resolve().parents[1] / "30.src" / "utils" / "utils.py"
spec = importlib.util.spec_from_file_location("jr_utils", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fill_pass_time_cells = module.fill_pass_time_cells
build_pass_time_weather_features = module.build_pass_time_weather_features
add_region_time_weather_features = module.add_region_time_weather_features


def test_single_pass_is_midpoint():
    """1回の通過は左右の時刻の中点で埋めることを確認する。"""
    s = pd.Series(["6:00", "通過", "6:30"])
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:15", "6:30"])
    pd.testing.assert_series_equal(result, expected)


def test_multiple_passes_are_evenly_split():
    """連続した通過セルは等間隔で補間されることを確認する。"""
    s = pd.Series(["6:00", "通過", "通過", "7:00"])
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:20", "6:40", "7:00"])
    pd.testing.assert_series_equal(result, expected)


def test_non_pass_values_are_preserved():
    """通過ではない値はそのまま保持されることを確認する。"""
    s = pd.Series(["6:00", "通過", "7:00", "7:20"]) 
    result = fill_pass_time_cells(s)
    expected = pd.Series(["6:00", "6:30", "7:00", "7:20"])
    pd.testing.assert_series_equal(result, expected)


def test_station_pass_weather_features_use_pass_hour():
    """駅の通過時刻と同じ時間帯の気象値を対応付けることを確認する。"""
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


def test_region_time_weather_features_are_added_from_document_formulas():
    """地域と時間帯ごとの降雪量・日射量・天気期待値を生成することを確認する。"""
    weather = pd.DataFrame(
        {
            "年月日時": pd.to_datetime([
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 00:00",
                "2024-01-01 01:00",
            ]),
            "地点": ["富山", "富山", "金沢", "金沢"],
            "気温(℃)": [0.0, 2.0, 1.0, 3.0],
            "降水量(mm)": [10.0, 20.0, 30.0, 40.0],
            "日照時間( 時間)": [0.0, 1.0, 0.5, 1.5],
            "天気": [1, 2, 1, 2],
        }
    )

    result = add_region_time_weather_features(weather)

    assert "富山_降雪量_0_00" in result.columns
    assert "富山_日射量_0_00" in result.columns
    assert "富山_天気期待値_0_00" in result.columns
    assert result.loc[(result["地点"] == "富山") & (result["時刻"] == 0), "富山_降雪量_0_00"].iat[0] > 0
    assert result.loc[(result["地点"] == "富山") & (result["時刻"] == 0), "富山_日射量_0_00"].iat[0] >= 0
