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
    def __init__(self):
        self.feature_importance_ = [3, 1, 2]


def test_time_series_folds_return_expected_count():
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
    model = DummyModel()
    output_path = tmp_path / "feature_importance.csv"

    saved = save_feature_importance(model, ["a", "b", "c"], output_path)

    assert output_path.exists()
    assert list(saved["feature"]) == ["a", "c", "b"]


def test_setup_logger_creates_timestamped_log_and_removes_stale_fixed_log(tmp_path: Path):
    stale_log = tmp_path / "pipeline.log"
    stale_log.write_text("legacy log content", encoding="utf-8")

    logger = setup_logger(stale_log)
    logger.info("hello timestamped log")

    assert not stale_log.exists()
    timestamped_logs = sorted(tmp_path.glob("pipeline_*.log"))
    assert len(timestamped_logs) == 1
    assert "hello timestamped log" in timestamped_logs[0].read_text(encoding="utf-8")


def test_cv_evaluation_logs_fold_metrics(caplog):
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
