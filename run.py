"""JR_train_snowの実行入口。

このモジュールはコマンドライン引数を受け取り、学習・CV・推論・提出ファイル生成を
一つのパイプラインとして制御する役割を持つ。
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "30.src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jr_snow.config import build_feature_columns, load_project_config
from jr_snow.cross_validation import time_series_folds
from jr_snow.data import load_train_test_data
from jr_snow.evaluation import compute_wmae, summarize_prediction_stats, summarize_target_stats
from jr_snow.feature_importance import save_feature_importance
from jr_snow.features import prepare_model_inputs
from jr_snow.logging_utils import setup_logger
from jr_snow.model_registry import save_model_artifact
from jr_snow.modeling import predict_submission, save_submission, train_lightgbm_model
from jr_snow.reporting import save_validation_report


def parse_args() -> argparse.Namespace:
    """実行時に指定された引数を解釈して、処理モードを決める。

    Returns:
        argparse.Namespace: どの処理を実行するかを表す設定値。
    """
    parser = argparse.ArgumentParser(description="JR_train_snow training and inference pipeline")
    # どの工程を実行するかを選ぶ。cvは交差検証、trainは学習、predictは予測、fullは両方を実施。
    parser.add_argument("--mode", choices=["train", "predict", "full", "cv"], default="full")
    parser.add_argument("--config", type=str, default=str(ROOT / "00.config" / "config.yaml"))
    parser.add_argument("--path-config", type=str, default=str(ROOT / "00.config" / "path.yaml"))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=str(ROOT / "artifacts" / "lightgbm_model.pkl"))
    parser.add_argument("--output-path", type=str, default=str(ROOT / "submit.csv"))
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=None)
    return parser.parse_args()


def resolve_cv_folds(cv_folds: int | None, config: dict) -> int:
    """CV fold数をCLI指定→設定ファイル順で解決する。

    時系列CVでは、コマンドラインで一時的に値を変えたい場合がある一方で、
    実験設定を YAML にまとめて管理したい場合もある。そこで、CLI が未指定なら
    config の値を使うようにして、実験条件と実行条件を分離する。
    """
    if cv_folds is not None:
        return cv_folds

    cv_config = config.get("CV", {}) if isinstance(config, dict) else {}
    value = cv_config.get("n_splits")
    if isinstance(value, int) and value >= 2:
        return value
    return 3


def evaluate_cv_folds(
    df: pd.DataFrame,
    feature_columns: dict,
    model_params: dict,
    n_splits: int = 3,
    logger: logging.Logger | None = None,
    target_col: str = "合計",
) -> dict[str, float | list[float]]:
    """時系列CVを実行して、各foldと平均WMAEを記録する。

    1. 時系列データを日付順に整列する
    2. foldごとにtrain/validを分割する
    3. LightGBMで学習してWMAEを算出する
    4. すべてのfoldの結果をログに出力して平均値を返す
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    working_df = df.copy()
    if "年月日" in working_df.columns:
        working_df["年月日"] = pd.to_datetime(working_df["年月日"])
    working_df = working_df.sort_values("年月日").reset_index(drop=True)

    # 追加で作成した閾値特徴量を含めて最終的な学習特徴量を組み立てる。
    feature_list = list(feature_columns.get("feature_list", []))
    engineered_cols = [
        column for column in working_df.columns if column not in feature_list and "ge_5.0_C" in column
    ]
    feature_list = list(dict.fromkeys(feature_list + engineered_cols))
    fix_columns = feature_list + ["年月日", target_col]
    working_df = working_df.reindex(columns=fix_columns)

    categorical_cols = [
        column for column in feature_columns.get("categorical_cols", []) if column in feature_list
    ]
    folds = time_series_folds(working_df, n_splits=n_splits, target_col=target_col)
    logger.info(f"CV fold count    : {len(folds)}")

    fold_wmae: list[float] = []
    for fold_index, (X_train, X_valid, y_train, y_valid) in enumerate(folds, start=1):
        # 各foldで学習に使うデータを必要列だけ取り出す。
        X_train = X_train[feature_list].copy()
        X_valid = X_valid[feature_list].copy()

        for column in feature_list:
            if column in categorical_cols:
                if column in X_train.columns:
                    X_train[column] = X_train[column].astype("category")
                    X_valid[column] = X_valid[column].astype("category")
                continue
            if column in X_train.columns:
                X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
                X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce")

        model = train_lightgbm_model(
            X_train,
            X_valid,
            y_train,
            y_valid,
            categorical_cols,
            model_params,
            num_boost_round=200,
            early_stopping_rounds=20,
        )
        valid_pred = model.predict(X_valid)
        fold_wmae_value = compute_wmae(y_valid, valid_pred)
        fold_wmae.append(fold_wmae_value)
        logger.info(f"CV fold {fold_index} WMAE   : {fold_wmae_value:.6f}")

    mean_wmae = sum(fold_wmae) / len(fold_wmae) if fold_wmae else 0.0
    logger.info(f"CV mean WMAE     : {mean_wmae:.6f}")
    return {"fold_wmae": fold_wmae, "mean_wmae": mean_wmae}


def main() -> None:
    """パイプラインの全体制御を行う。

    1. 設定ファイルを読み込む
    2. 学習データとテストデータを読む
    3. 特徴量を整形する
    4. モード別に学習、CV、予測、提出ファイル生成を実行する
    """
    args = parse_args()
    logger = setup_logger(ROOT / "logs" / "pipeline.log")

    config, path_config = load_project_config(args.config, args.path_config)
    feature_columns = build_feature_columns(config)
    path_map = path_config.get("INTERIUM_PATH", path_config)
    experiment_note = config.get("EXPERIMENT", {}).get("experiment_note", "特徴量・評価の整理を実施")
    cv_folds = resolve_cv_folds(args.cv_folds, config)

    logger.info("=== JR_train_snow pipeline start ===")
    logger.info(f"Experiment       : {experiment_note}")
    logger.info("Implementation   : feature engineering, validation, reporting, and model registry are separated into modules.")

    if args.mode == "cv":
        # CV専用の処理は、学習データだけを使ってfoldごとの評価を実施する。
        train_df, _ = load_train_test_data(
            train_data_path=args.train_data or path_map.get("train_data"),
            test_data_path=args.test_data or path_map.get("test_data"),
            path_config=path_config,
        )
        evaluate_cv_folds(
            train_df,
            feature_columns,
            config["MODEL_PARAMS"],
            n_splits=cv_folds,
            logger=logger,
        )
        return

    # 学習・予測の共通前処理はここで行う。train/testのロードと特徴量整形をまとめて実行する。
    train_df, test_df = load_train_test_data(
        train_data_path=args.train_data or path_map.get("train_data"),
        test_data_path=args.test_data or path_map.get("test_data"),
        path_config=path_config,
    )

    prepared = prepare_model_inputs(
        df_train=train_df,
        df_test=test_df,
        feature_columns=feature_columns,
        split_date=config["split_date"],
    )

    logger.info(f"Train件数         : {len(prepared['X_train']) + len(prepared['y_train'])}")
    logger.info(f"Validation件数    : {len(prepared['X_valid']) + len(prepared['y_valid'])}")
    logger.info(f"Test件数          : {len(prepared['df_test_processed'])}")
    logger.info(f"特徴量数          : {len(prepared['feature_list'])}")
    logger.info(f"Target名         : 合計")
    logger.info(f"使用特徴量        : {prepared['feature_list']}")
    logger.info(f"LightGBMパラメータ: {config['MODEL_PARAMS']}")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode in {"train", "full"}:
        # 学習の実行後、モデルと評価結果をアーティファクトとして保存する。
        model = train_lightgbm_model(
            prepared["X_train"],
            prepared["X_valid"],
            prepared["y_train"],
            prepared["y_valid"],
            prepared["categorical_cols"],
            config["MODEL_PARAMS"],
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model_version_path = save_model_artifact(model, ROOT / "artifacts", prefix="lightgbm_model")
        save_feature_importance(model, prepared["feature_list"], ROOT / "artifacts" / "feature_importance.csv")
        save_validation_report(
            {
                "valid_wmae": compute_wmae(prepared["y_valid"], model.predict(prepared["X_valid"])),
                "train_rows": len(prepared["X_train"]),
                "valid_rows": len(prepared["X_valid"]),
                "test_rows": len(prepared["df_test_processed"]),
                "feature_count": len(prepared["feature_list"]),
            },
            ROOT / "artifacts" / "validation_report.csv",
        )
        logger.info(f"Training result   : best_iteration={getattr(model, 'best_iteration', 'n/a')}, best_score={model.best_score}")
        logger.info(f"Model saved      : {model_version_path}")
    else:
        with model_path.open("rb") as file:
            model = pickle.load(file)
        logger.info(f"Model loaded     : {model_path}")

    if args.mode in {"predict", "full"}:
        # 推論後に提出用CSVを作成し、検証WMAEも併せてログ出力する。
        predictions = predict_submission(model, prepared["df_test_processed"], prepared["feature_list"])
        save_submission(predictions, output_path=args.output_path)
        valid_wmae = compute_wmae(prepared["y_valid"], model.predict(prepared["X_valid"]))
        prediction_stats = summarize_prediction_stats(predictions)
        target_stats = summarize_target_stats(prepared["y_train"])

        logger.info(
            f"Prediction stats : "
            f"min={prediction_stats['min']:.4f}, "
            f"max={prediction_stats['max']:.4f}, "
            f"mean={prediction_stats['mean']:.4f}"
        )
        logger.info(
            f"Target stats     : "
            f"min={target_stats['min']:.4f}, "
            f"max={target_stats['max']:.4f}, "
            f"mean={target_stats['mean']:.4f}"
        )
        logger.info(f"Validation WMAE   : {valid_wmae:.6f}")
        logger.info(f"Submission saved : {args.output_path}")
        logger.info(f"Submit model validation WMAE : {valid_wmae:.6f}")


if __name__ == "__main__":
    main()
