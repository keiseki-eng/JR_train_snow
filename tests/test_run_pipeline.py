"""設定ファイルと特徴量定義の基本動作を確認するテスト。

YAMLから設定を読み取り、特徴量リストが生成できることを確認する。
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "30.src"))

from jr_snow.config import build_feature_columns, load_yaml_config
from jr_snow.features import fill_missing_by_same_day_time_location_average, prepare_model_inputs


def test_config_and_feature_columns_are_generated():
    """設定ファイルから特徴量定義が正しく生成されることを確認する。"""
    config = load_yaml_config(ROOT / "00.config" / "config.yaml")
    feature_columns = build_feature_columns(config)

    assert config["split_date"] == "2016-12-10"
    assert "列車番号" in feature_columns["base_features"]
    assert len(feature_columns["weather_features"]) > 0
    assert len(feature_columns["snow_features"]) > 0


def test_prepare_model_inputs_adds_document_weather_features_by_default():
    """通常の学習前処理で、地域・時間ごとの降雪量・日射量・天気期待値が導入されることを確認する。"""
    config = load_yaml_config(ROOT / "00.config" / "config.yaml")
    feature_columns = build_feature_columns(config)

    train_df = pd.DataFrame(
        {
            "年月日": ["2024-01-01", "2024-01-02"],
            "列車番号": ["1", "2"],
            "地点": ["富山", "富山"],
            "年月日時": ["2024-01-01 00:00", "2024-01-02 00:00"],
            "気温(℃)": [0.0, 2.0],
            "降水量(mm)": [10.0, 20.0],
            "日照時間( 時間)": [0.0, 1.0],
            "天気": [1, 2],
            "合計": [10.0, 15.0],
        }
    )
    test_df = train_df.copy()

    prepared = prepare_model_inputs(train_df, test_df, feature_columns, split_date="2024-01-02", target_col="合計")

    assert "富山_降雪量_0_00" in prepared["feature_list"]
    assert "富山_日射量_0_00" in prepared["feature_list"]
    assert "富山_天気期待値_0_00" in prepared["feature_list"]
    assert prepared["X_train"]["富山_降雪量_0_00"].notna().any()
    assert prepared["X_train"]["富山_日射量_0_00"].notna().any()
    assert prepared["X_train"]["富山_天気期待値_0_00"].notna().any()


def test_fill_missing_by_same_day_time_location_average_uses_other_locations():
    """同じ日・同じ時間帯の他地点平均で欠損値を補完することを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"],
            "地点": ["富山", "金沢", "福井", "富山"],
            "年月日時": [
                "2024-01-01 00:00",
                "2024-01-01 00:00",
                "2024-01-01 00:00",
                "2024-01-01 01:00",
            ],
            "気温(℃)": [None, 2.0, 4.0, 5.0],
            "降水量(mm)": [10.0, 20.0, None, 30.0],
        }
    )

    filled = fill_missing_by_same_day_time_location_average(df)

    assert filled.loc[0, "気温(℃)"] == 3.0
    assert filled.loc[2, "降水量(mm)"] == 15.0
