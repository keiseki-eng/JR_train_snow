"""JR_train_snow の機械学習パイプラインを構成するモジュール群。

設定読み込み、特徴量作成、CV、学習、評価、ログ出力、提出ファイル生成までの
工程を一つのパッケージとしてまとめている。
"""

from .config import build_feature_columns, load_project_config, load_yaml_config
from .cross_validation import time_series_folds
from .data import load_path_config, load_train_test_data
from .evaluation import (
    compute_roc_auc,
    compute_roc_curve,
    compute_wmae,
    plot_roc_curve,
    summarize_prediction_stats,
    summarize_target_stats,
)
from .feature_importance import save_feature_importance
from .features import prepare_model_inputs
from .logging_utils import setup_logger
from .model_registry import build_model_version, load_model_artifact, save_model_artifact
from .modeling import train_lightgbm_model, wmae_eval
from .reporting import save_validation_report

__all__ = [
    "build_feature_columns",
    "load_project_config",
    "load_yaml_config",
    "load_path_config",
    "load_train_test_data",
    "prepare_model_inputs",
    "train_lightgbm_model",
    "wmae_eval",
    "compute_wmae",
    "compute_roc_auc",
    "compute_roc_curve",
    "plot_roc_curve",
    "summarize_prediction_stats",
    "summarize_target_stats",
    "setup_logger",
    "time_series_folds",
    "save_feature_importance",
    "save_model_artifact",
    "load_model_artifact",
    "build_model_version",
    "save_validation_report",
]
