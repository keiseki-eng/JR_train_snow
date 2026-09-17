"""設定ファイルと特徴量定義の基本動作を確認するテスト。

YAMLから設定を読み取り、特徴量リストが生成できることを確認する。
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "30.src"))

from jr_snow.config import build_feature_columns, load_yaml_config


def test_config_and_feature_columns_are_generated():
    """設定ファイルから特徴量定義が正しく生成されることを確認する。"""
    config = load_yaml_config(ROOT / "00.config" / "config.yaml")
    feature_columns = build_feature_columns(config)

    assert config["split_date"] == "2016-12-10"
    assert "列車番号" in feature_columns["base_features"]
    assert len(feature_columns["weather_features"]) > 0
    assert len(feature_columns["snow_features"]) > 0
