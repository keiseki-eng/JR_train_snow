"""CV とログ・重要度出力を確認する回帰テスト。

特徴量生成やログのタイムスタンプ付与が意図どおりに動くかを検証する。
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run
from jr_snow.cross_validation import time_series_folds
from jr_snow.feature_importance import save_feature_importance
from jr_snow.logging_utils import setup_logger


class DummyModel:
    """特徴量重要度を持つ簡易モデル。"""

    def __init__(self):
        self.feature_importance_ = [3, 1, 2]


def test_time_series_folds_return_expected_count():
    """time_series_folds が指定数のfoldを返すことを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=12, freq="D"),
            "合計": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "x": list(range(12)),
        }
    )

    folds = time_series_folds(df, n_splits=3)
    assert len(folds) == 3
    assert all(len(valid) > 0 for _, valid, _, _ in folds)


def test_feature_importance_csv_is_created(tmp_path: Path):
    """重要度をCSVに保存できることを確認する。"""
    model = DummyModel()
    output_path = tmp_path / "feature_importance.csv"

    saved = save_feature_importance(model, ["a", "b", "c"], output_path)

    assert output_path.exists()
    assert list(saved["feature"]) == ["a", "c", "b"]


def test_setup_logger_creates_timestamped_log_and_removes_stale_fixed_log(tmp_path: Path):
    """古い固定ログを削除し、日時付きログのみが残ることを確認する。"""
    stale_log = tmp_path / "pipeline.log"
    stale_log.write_text("legacy log content", encoding="utf-8")

    logger = setup_logger(stale_log)
    logger.info("hello timestamped log")

    assert not stale_log.exists()
    timestamped_logs = sorted(tmp_path.glob("pipeline_*.log"))
    assert len(timestamped_logs) == 1
    assert "hello timestamped log" in timestamped_logs[0].read_text(encoding="utf-8")


def test_cv_evaluation_logs_fold_metrics(caplog):
    """CVの各fold結果と平均WMAEがログに出力されることを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=30, freq="D"),
            "合計": list(range(30)),
            "x": [i % 3 for i in range(30)],
            "year": 2024,
        }
    )
    feature_columns = {
        "feature_list": ["year", "x"],
        "categorical_cols": ["year"],
    }

    logger = logging.getLogger("cv_test")
    with caplog.at_level(logging.INFO, logger="cv_test"):
        metrics = run.evaluate_cv_folds(
            df,
            feature_columns,
            {"objective": "regression", "metric": "l2", "verbose": -1},
            n_splits=2,
            logger=logger,
        )

    assert len(metrics["fold_wmae"]) == 2
    assert any("CV fold 1 WMAE" in record.message for record in caplog.records)
    assert any("CV mean WMAE" in record.message for record in caplog.records)


def test_resolve_cv_folds_prefers_cli_value_then_config():
    """CV fold数はCLI指定を優先し、未指定時は設定ファイル値を使う。"""
    config = {"CV": {"n_splits": 5}}
    assert run.resolve_cv_folds(None, config) == 5
    assert run.resolve_cv_folds(2, config) == 2
