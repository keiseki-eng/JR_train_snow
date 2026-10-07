"""CV とログ・重要度出力を確認する回帰テスト。

特徴量生成やログのタイムスタンプ付与が意図どおりに動くかを検証する。
"""

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run
from jr_snow.cross_validation import time_series_folds
from jr_snow.evaluation import compute_roc_auc
from jr_snow.feature_importance import save_feature_importance, save_shap_summary_plot
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


def test_time_series_folds_use_expanding_window_and_expand_training_periods():
    """time_series_folds が train を広げていく expanding-window CV を返すことを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=30, freq="D"),
            "合計": list(range(30)),
            "x": list(range(30)),
        }
    )

    folds = time_series_folds(df, n_splits=3)

    assert len(folds) == 3
    train_end_dates = [train_df["年月日"].max() for train_df, _, _, _ in folds]
    valid_start_dates = [valid_df["年月日"].min() for _, valid_df, _, _ in folds]
    assert train_end_dates[0] < valid_start_dates[0]
    assert train_end_dates[1] < valid_start_dates[1]
    assert train_end_dates[2] < valid_start_dates[2]
    assert train_end_dates[0] < train_end_dates[1] < train_end_dates[2]


def test_feature_importance_csv_is_created(tmp_path: Path):
    """重要度をCSVに保存できることを確認する。"""
    model = DummyModel()
    output_path = tmp_path / "feature_importance.csv"

    saved = save_feature_importance(model, ["a", "b", "c"], output_path)

    assert output_path.exists()
    assert list(saved["feature"]) == ["a", "c", "b"]


def test_save_shap_summary_plot_creates_images(tmp_path: Path):
    """SHAPの要約図を画像として保存できることを確認する。"""
    class DummyTreeModel:
        def __init__(self):
            self.n_features_in_ = 3

    import shap

    class DummyExplainer:
        def __init__(self, model, data):
            self.model = model
            self.data = data

        def shap_values(self, X):
            values = np.zeros((len(X), X.shape[1]))
            return values

    with patch("jr_snow.feature_importance.shap.TreeExplainer", return_value=DummyExplainer(None, None)):
        X = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["a", "b", "c"])
        output_dir = tmp_path / "shap"

        paths = save_shap_summary_plot(DummyTreeModel(), X, output_dir)

    assert output_dir.joinpath("shap_summary.png").exists()
    assert output_dir.joinpath("shap_waterfall_0.png").exists()
    assert len(paths) == 2


def test_parse_args_accepts_final_inference_strategy():
    """実行時CLIで最終推論戦略を選べることを確認する。"""
    with patch.object(sys, "argv", ["run.py", "--mode", "full", "--final-model-strategy", "cv_average"]):
        args = run.parse_args()
    assert args.final_model_strategy == "cv_average"


def test_parse_args_accepts_two_stage_snow_prediction_flag():
    """着雪量の2段階予測をCLIで有効化できることを確認する。"""
    with patch.object(sys, "argv", ["run.py", "--mode", "predict", "--two-stage-snow-prediction"]):
        args = run.parse_args()
    assert args.two_stage_snow_prediction is True


def test_apply_two_stage_snow_prediction_zeroes_non_target_rows():
    """着雪量予測フラグが0の行は0を返し、1の行だけモデル予測を使うことを確認する。"""
    predictions = np.array([10.0, 20.0, 30.0])
    flags = pd.Series([0, 1, 0], name="着雪量予測フラグ")

    gated = run.apply_two_stage_snow_prediction(predictions, flags)

    assert gated.tolist() == [0.0, 20.0, 0.0]


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


def test_compute_roc_auc_returns_expected_score():
    """ROC-AUC がしきい値判定ではなく確率スコアで計算されることを確認する。"""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    auc = compute_roc_auc(y_true, y_score)

    assert auc == 0.75


def test_filter_snow_presence_records_uses_threshold_for_binary_flag():
    """着雪有無フラグがしきい値以上の行だけを残すことを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=5, freq="D"),
            "着雪有無フラグ": [0, 1, 0, 1, 0],
            "合計": [0.0, 1.0, 0.0, 2.0, 0.0],
        }
    )

    filtered = run.filter_snow_presence_records(df, thresholds=0.005)

    assert len(filtered) == 2
    assert filtered["着雪有無フラグ"].tolist() == [1, 1]


def test_filter_snow_presence_records_uses_probability_column_when_present():
    """着雪確率カラムがある場合は、確率値に対してしきい値フィルタが効くことを確認する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=4, freq="D"),
            "着雪確率": [0.001, 0.005, 0.02, 0.0],
            "合計": [0.1, 0.2, 0.5, 0.0],
        }
    )

    filtered = run.filter_snow_presence_records(df, thresholds=0.005)

    assert len(filtered) == 2
    assert filtered["着雪確率"].tolist() == [0.005, 0.02]


def test_filter_snow_presence_records_requires_winter_flag_before_thresholding():
    """冬季フラグが存在する場合は、冬季レコードに限定してからしきい値を適用する。"""
    df = pd.DataFrame(
        {
            "年月日": pd.date_range("2024-01-01", periods=5, freq="D"),
            "冬季フラグ": [0, 1, 1, 1, 0],
            "着雪確率": [0.9, 0.001, 0.02, 0.03, 0.8],
            "合計": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )

    filtered = run.filter_snow_presence_records(df, thresholds=0.005)

    assert filtered["冬季フラグ"].tolist() == [1, 1]
    assert filtered["着雪確率"].tolist() == [0.02, 0.03]


def test_apply_two_stage_snow_prediction_uses_winter_flag_and_probability_threshold():
    """冬季外の行は0、冬季内でも確率閾値未満は0とする。"""
    predictions = np.array([10.0, 20.0, 30.0, 40.0])
    winter_flags = pd.Series([0, 1, 1, 0], name="冬季フラグ")
    snow_probabilities = pd.Series([0.9, 0.9, 0.1, 0.9], name="着雪確率")

    gated = run.apply_two_stage_snow_prediction(
        predictions,
        winter_flags,
        snow_probabilities=snow_probabilities,
        thresholds=0.5,
    )

    assert gated.tolist() == [0.0, 20.0, 0.0, 0.0]
