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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "30.src"
DEFAULT_INTERMEDIATE_DATA_PATH = ROOT / "artifacts" / "intermediate" / "prepared_dataset.pkl"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jr_snow.config import build_feature_columns, load_project_config
from jr_snow.cross_validation import time_series_folds
from jr_snow.data import load_train_test_data
from jr_snow.evaluation import compute_roc_auc, compute_wmae, summarize_prediction_stats, summarize_target_stats
from jr_snow.feature_importance import save_feature_importance, save_shap_summary_plot
from jr_snow.features import add_document_weather_features, build_engineered_feature_frame, prepare_model_inputs
from jr_snow.logging_utils import setup_logger
from jr_snow.model_registry import save_model_artifact
from jr_snow.modeling import (
    apply_prediction_correction,
    predict_submission,
    save_submission,
    train_lightgbm_model,
    train_lightgbm_model_on_full_data,
)
from jr_snow.reporting import save_validation_report



def parse_args() -> argparse.Namespace:
    """実行時に指定された引数を解釈して、処理モードを決める。

    Returns:
        argparse.Namespace: どの処理を実行するかを表す設定値。
    """
    parser = argparse.ArgumentParser(description="JR_train_snow training and inference pipeline")
    # どの工程を実行するかを選ぶ。prepare は特徴量生成済みデータセットを保存し、
    # train / predict / full はその中間データを再利用する。
    parser.add_argument("--mode", choices=["prepare", "train", "predict", "full", "cv"], default="full")
    parser.add_argument("--config", type=str, default=str(ROOT / "00.config" / "config.yaml"))
    parser.add_argument("--path-config", type=str, default=str(ROOT / "00.config" / "path.yaml"))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--intermediate-data", type=str, default=None, help="特徴量生成後の中間データセットを保存/再利用するpickleファイル")
    parser.add_argument("--model-path", type=str, default=str(ROOT / "artifacts" / "lightgbm_model.pkl"))
    parser.add_argument("--output-path", type=str, default=str(ROOT / "submit.csv"))
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument(
        "--final-model-strategy",
        choices=["single_split", "cv_average", "median_wmae", "best_fold"],
        default="single_split",
        help=(
            "最終推論で使うモデル戦略: single_split は通常の学習/検証分割, "
            "cv_average は各CVモデル予測の平均, median_wmae は中央値に近いモデルを選択, "
            "best_fold はWMAE最良のfoldモデルを採用"
        ),
    )
    parser.add_argument(
        "--two-stage-snow-prediction",
        action="store_true",
        help=(
            "テストデータの '着雪量予測フラグ' が 0 の行は予測結果を 0 にし、"
            "1 の行だけモデル予測を使う二段階予測を有効化する。"
        ),
    )
    parser.add_argument(
        "--skip-final-training",
        action="store_true",
        help=(
            "最終学習を実行せず、既存のモデルファイルを再利用する。"
            "既定では全学習データを用いた最終学習を実行する。"
        ),
    )
    return parser.parse_args()


def filter_snow_presence_records(
    df: pd.DataFrame,
    thresholds: float | int = 0.0,
    candidate_columns: tuple[str, ...] = ("着雪有無フラグ", "着雪有無", "着雪確率"),
    winter_flag_column: str = "冬季フラグ",
) -> pd.DataFrame:
    """冬季レコードに限定したうえで、着雪確率が閾値以上の行だけを残す。

    学習時に冬季外のデータを混ぜないようにし、着雪量モデルは冬季かつ
    着雪確率が threshold 以上のレコードにのみ当てはめる。
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()
    if winter_flag_column in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered[winter_flag_column], errors="coerce") == 1].copy()

    threshold_value = float(thresholds)
    for column in candidate_columns:
        if column not in filtered.columns:
            continue
        filtered = filtered.loc[pd.to_numeric(filtered[column], errors="coerce") >= threshold_value].copy()
        return filtered

    return filtered


def apply_two_stage_snow_prediction(
    predictions: np.ndarray | list[float],
    snow_prediction_flags: pd.Series | None,
    snow_probabilities: pd.Series | None = None,
    thresholds: float | int = 0.0,
) -> np.ndarray:
    """冬季フラグと着雪確率の両方を考慮して、対象外の行を 0 に置き換える。"""
    prediction_array = np.asarray(predictions, dtype=float)
    if snow_prediction_flags is None:
        return prediction_array

    flags = pd.to_numeric(snow_prediction_flags, errors="coerce").fillna(0).astype(int)
    if len(flags) != len(prediction_array):
        raise ValueError(
            "Two-stage snow prediction requires the flag length to match prediction length: "
            f"flags={len(flags)}, predictions={len(prediction_array)}"
        )

    gated_predictions = np.where(flags == 1, prediction_array, 0.0).astype(float)
    if snow_probabilities is None:
        return gated_predictions

    probabilities = pd.to_numeric(snow_probabilities, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(probabilities) != len(prediction_array):
        raise ValueError(
            "Two-stage snow prediction requires the probability length to match prediction length: "
            f"probabilities={len(probabilities)}, predictions={len(prediction_array)}"
        )

    threshold_value = float(thresholds)
    if not np.any(probabilities >= threshold_value):
        return prediction_array

    valid_mask = (flags == 1) & (probabilities >= threshold_value)
    return np.where(valid_mask, prediction_array, 0.0).astype(float)


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


def resolve_intermediate_data_path(intermediate_path: str | None, path_config: dict) -> Path:
    """中間データセットの保存先をCLI・設定ファイル・デフォルト順で決める。"""
    if intermediate_path is not None:
        return Path(intermediate_path)

    path_map = path_config.get("INTERIUM_PATH", path_config)
    for key in ("intermediate_data", "prepared_data", "feature_data", "feature_engineered_data"):
        value = path_map.get(key)
        if value:
            return Path(value)
    return DEFAULT_INTERMEDIATE_DATA_PATH


def save_intermediate_dataset(prepared: dict[str, object], output_path: str | Path) -> Path:
    """特徴量生成後の学習・検証・推論データをpickleとして保存する。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(prepared, file)
    return path


def load_intermediate_dataset(path: str | Path) -> dict[str, object]:
    """中間データセットを読み込んで学習/推論へ再利用する。"""
    with Path(path).open("rb") as file:
        prepared = pickle.load(file)
    if not isinstance(prepared, dict):
        raise TypeError(f"Intermediate dataset is not a dictionary: {path}")
    return prepared


def run_cv_models(
    df: pd.DataFrame,
    feature_columns: dict,
    model_params: dict,
    n_splits: int = 3,
    logger: logging.Logger | None = None,
    target_col: str = "合計",
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> dict[str, list | float | dict]:
    """expanding-window CV を回し、各 fold の学習済みモデルと WMAE を返す。"""
    if logger is None:
        logger = logging.getLogger(__name__)

    working_df = df.copy()
    working_df = add_document_weather_features(working_df)
    if "年月日" in working_df.columns:
        working_df["年月日"] = pd.to_datetime(working_df["年月日"])
    working_df = working_df.sort_values("年月日").reset_index(drop=True)

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

    fold_results: list[dict[str, float | int | object]] = []
    for fold_index, (X_train, X_valid, y_train, y_valid) in enumerate(folds, start=1):
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
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        valid_pred = model.predict(X_valid)
        fold_wmae_value = compute_wmae(y_valid, valid_pred)
        fold_roc_auc = None
        try:
            fold_roc_auc = compute_roc_auc(y_valid, valid_pred)
        except ValueError:
            pass

        fold_results.append({
            "fold_index": fold_index,
            "model": model,
            "wmae": fold_wmae_value,
            "roc_auc": fold_roc_auc,
            "valid_pred": valid_pred,
        })
        logger.info(f"CV fold {fold_index} WMAE   : {fold_wmae_value:.6f}")
        if fold_roc_auc is not None:
            logger.info(f"CV fold {fold_index} ROC-AUC: {fold_roc_auc:.6f}")

    mean_wmae = sum(item["wmae"] for item in fold_results) / len(fold_results) if fold_results else 0.0
    mean_roc_auc = (
        sum(item["roc_auc"] for item in fold_results if item["roc_auc"] is not None) / len([item for item in fold_results if item["roc_auc"] is not None])
        if any(item["roc_auc"] is not None for item in fold_results)
        else None
    )
    logger.info(f"CV mean WMAE     : {mean_wmae:.6f}")
    if mean_roc_auc is not None:
        logger.info(f"CV mean ROC-AUC  : {mean_roc_auc:.6f}")
    return {
        "fold_results": fold_results,
        "mean_wmae": mean_wmae,
        "mean_roc_auc": mean_roc_auc,
        "fold_wmae": [item["wmae"] for item in fold_results],
        "fold_roc_auc": [item["roc_auc"] for item in fold_results],
    }


def evaluate_cv_folds(
    df: pd.DataFrame,
    feature_columns: dict,
    model_params: dict,
    n_splits: int = 3,
    logger: logging.Logger | None = None,
    target_col: str = "合計",
) -> dict[str, float | list[float] | None]:
    """時系列CVを実行して、各foldと平均WMAEを記録する。"""
    results = run_cv_models(
        df=df,
        feature_columns=feature_columns,
        model_params=model_params,
        n_splits=n_splits,
        logger=logger,
        target_col=target_col,
    )
    return {
        "fold_wmae": results["fold_wmae"],
        "mean_wmae": results["mean_wmae"],
        "fold_roc_auc": results["fold_roc_auc"],
        "mean_roc_auc": results["mean_roc_auc"],
    }



def train_final_model_on_all_data(
    df: pd.DataFrame,
    feature_columns: dict,
    model_params: dict,
    *,
    target_col: str = "合計",
    num_boost_round: int = 1000,
    logger: logging.Logger | None = None,
    feature_list: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    threshold: float = 5.0,
    reference_frame: pd.DataFrame | None = None,
):
    """全学習データで最終モデルを学習する。

    学習データと検証データでカテゴリ列・カテゴリ定義を統一する。
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    resolved_feature_list = list(
        feature_list or feature_columns.get("feature_list", [])
    )
    configured_categorical_cols = list(
        categorical_cols or feature_columns.get("categorical_cols", [])
    )

    working_df = build_engineered_feature_frame(
        df.copy(),
        threshold=threshold,
    )

    if "年月日" in working_df.columns:
        working_df["年月日"] = pd.to_datetime(
            working_df["年月日"]
        )

    working_df = working_df.sort_values(
        "年月日"
    ).reset_index(drop=True)

    fix_columns = list(
        dict.fromkeys(
            resolved_feature_list + ["年月日", target_col]
        )
    )
    working_df = working_df.reindex(columns=fix_columns)

    X = working_df[resolved_feature_list].copy()
    y = pd.to_numeric(
        working_df[target_col],
        errors="coerce",
    )

    # 参照用DataFrameはコピーして、元の検証データを不用意に変更しない。
    reference = None
    if reference_frame is not None:
        reference = reference_frame.reindex(
            columns=resolved_feature_list
        ).copy()

    # 設定指定のカテゴリ列に加えて、実際のDataFrameで
    # object/category型になっている列もカテゴリ列として検出する。
    category_columns = {
        column
        for column in configured_categorical_cols
        if column in X.columns
    }

    for frame in [X, reference]:
        if frame is None:
            continue

        category_columns.update(
            frame.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        )

    category_columns = sorted(
        category_columns.intersection(X.columns)
    )

    # 全学習データと検証データでカテゴリの候補を統合し、
    # 同じカテゴリ定義を使用する。
    for column in resolved_feature_list:
        if column in category_columns:
            all_values = []

            for frame in [X, reference]:
                if frame is not None and column in frame.columns:
                    all_values.extend(
                        frame[column]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

            categories = pd.Index(
                sorted(set(all_values))
            )

            # 欠損値は欠損値のまま保持し、カテゴリ定義だけを共有する。
            category_dtype = pd.CategoricalDtype(
                categories=categories,
                ordered=False,
            )

            X[column] = (
                X[column]
                .astype("string")
                .astype(category_dtype)
            )

            if reference is not None and column in reference.columns:
                reference[column] = (
                    reference[column]
                    .astype("string")
                    .astype(category_dtype)
                )

        else:
            # カテゴリ列以外は数値として扱う。
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            )

            if reference is not None and column in reference.columns:
                reference[column] = pd.to_numeric(
                    reference[column],
                    errors="coerce",
                )

    resolved_categorical_cols = category_columns

    # 学習直前にカテゴリ列の状態をログへ出力する。
    logger.info(
        "Final model retraining: rows=%d, features=%d",
        len(X),
        len(resolved_feature_list),
    )
    logger.info(
        "Final categorical columns (%d): %s",
        len(resolved_categorical_cols),
        resolved_categorical_cols,
    )

    logger.info(
        "Final training category dtypes: %s",
        {
            column: str(X[column].dtype)
            for column in resolved_categorical_cols
        },
    )

    # 全学習データで最終モデルを学習する。
    model = train_lightgbm_model_on_full_data(
        X,
        y,
        resolved_categorical_cols,
        model_params,
        num_boost_round=num_boost_round,
    )

    # カテゴリ定義を統一した検証データも呼び出し元へ返す。
    return model, reference


def select_model_for_final_inference(
    cv_results: dict[str, list | float | dict],
    strategy: str,
) -> tuple[str, object | None, float | None]:
    """最終推論に使うモデルを戦略に応じて選ぶ。"""
    fold_results = cv_results.get("fold_results", [])
    if not fold_results:
        return strategy, None, None

    if strategy == "single_split":
        return strategy, fold_results[0]["model"], float(fold_results[0]["wmae"])

    if strategy == "cv_average":
        return strategy, None, float(cv_results["mean_wmae"])

    if strategy == "median_wmae":
        wmae_values = np.asarray([item["wmae"] for item in fold_results], dtype=float)
        median_value = float(np.median(wmae_values))
        selected = min(fold_results, key=lambda item: abs(float(item["wmae"]) - median_value))
        return strategy, selected["model"], float(selected["wmae"])

    if strategy == "best_fold":
        selected = min(fold_results, key=lambda item: float(item["wmae"]))
        return strategy, selected["model"], float(selected["wmae"])

    raise ValueError(f"Unsupported final-model strategy: {strategy}")


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
    logger.info(f"Final model strategy : {args.final_model_strategy}")

    if args.mode == "cv":
        # CV専用の処理は、学習データだけを使ってfoldごとの評価を実施する。
        train_df, _ = load_train_test_data(
            train_data_path=args.train_data or path_map.get("train_data"),
            test_data_path=args.test_data or path_map.get("test_data"),
            path_config=path_config,
        )
        feature_list = list(feature_columns.get("feature_list", []))
        logger.info(f"使用特徴量        : {feature_list}")
        logger.info(f"使用特徴量(文字列): {', '.join(feature_list)}")
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

    thresholds = float(config.get("thresholds", 0.0))
    train_before_rows = len(train_df)
    if any(column in train_df.columns for column in ("冬季フラグ", "着雪有無フラグ", "着雪有無", "着雪確率")):
        train_df = filter_snow_presence_records(train_df, thresholds=thresholds)
        logger.info(
            "Snow presence filter: thresholds=%s, train rows %d -> %d",
            thresholds,
            train_before_rows,
            len(train_df),
        )

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    strategy = args.final_model_strategy
    logger.info(f"Final model strategy : {strategy}")
    logger.info(f"Final model strategy (CLI): {args.final_model_strategy}")

    intermediate_path = resolve_intermediate_data_path(args.intermediate_data, path_config)

    if args.mode == "prepare":
        logger.info(f"Feature dataset generation mode: {intermediate_path}")
        prepared = prepare_model_inputs(
            df_train=train_df,
            df_test=test_df,
            feature_columns=feature_columns,
            split_date=config["split_date"],
        )
        save_intermediate_dataset(prepared, intermediate_path)
        logger.info(f"Intermediate dataset saved: {intermediate_path}")
        logger.info(f"Train/valid/test rows: {len(prepared['X_train'])}, {len(prepared['X_valid'])}, {len(prepared['df_test_processed'])}")
        return
    
    
    if args.mode in {"train", "full"}:
        prepared = prepare_model_inputs(
            df_train=train_df,
            df_test=test_df,
            feature_columns=feature_columns,
            split_date=config["split_date"],
        )
        save_intermediate_dataset(prepared, intermediate_path)
        logger.info(f"Regenerated intermediate dataset for training/full mode: {intermediate_path}")
    elif intermediate_path.exists():
        prepared = load_intermediate_dataset(intermediate_path)
        logger.info(f"Using cached intermediate dataset: {intermediate_path}")
    else:
        prepared = prepare_model_inputs(
            df_train=train_df,
            df_test=test_df,
            feature_columns=feature_columns,
            split_date=config["split_date"],
        )
        save_intermediate_dataset(prepared, intermediate_path)
        logger.info(f"Intermediate dataset saved: {intermediate_path}")

    validation_X_for_prediction = prepared["X_valid"]

    logger.info(f"Train件数         : {len(prepared['X_train']) + len(prepared['y_train'])}")
    logger.info(f"Validation件数    : {len(prepared['X_valid']) + len(prepared['y_valid'])}")
    logger.info(f"Test件数          : {len(prepared['df_test_processed'])}")
    logger.info(f"特徴量数          : {len(prepared['feature_list'])}")
    logger.info(f"Target名         : 合計")
    logger.info(f"使用特徴量        : {prepared['feature_list']}")
    logger.info(f"使用特徴量(文字列): {', '.join(prepared['feature_list'])}")
    logger.info(f"LightGBMパラメータ: {config['MODEL_PARAMS']}")

    if args.mode in {"train", "full"}:
        if args.skip_final_training:
            if model_path.exists():
                with model_path.open("rb") as file:
                    model = pickle.load(file)
                logger.info(f"Model loaded     : {model_path}")
            else:
                raise FileNotFoundError(f"Skip-final-training is enabled but no model exists at {model_path}")
        else:
            final_model, final_X_valid = train_final_model_on_all_data(
                train_df,
                feature_columns,
                config["MODEL_PARAMS"],
                target_col="合計",
                num_boost_round=args.num_boost_round,
                logger=logger,
                feature_list=prepared["feature_list"],
                categorical_cols=prepared["categorical_cols"],
                threshold=thresholds,
                reference_frame=prepared["X_valid"],
            )
            validation_X_for_prediction = final_X_valid
            model_version_path = save_model_artifact(final_model, ROOT / "artifacts", prefix="lightgbm_model")
            save_feature_importance(final_model, prepared["feature_list"], ROOT / "artifacts" / "feature_importance.csv")
            try:
                save_shap_summary_plot(
                    final_model,
                    final_X_valid,
                    ROOT / "artifacts" / "shap",
                    categorical_cols=prepared["categorical_cols"],
                )
            except (ModuleNotFoundError, ValueError):
                logger.warning(
                    "SHAP plot generation failed for the final model; skipping plot export so submission generation can continue.",
                    exc_info=True,
                )
            save_validation_report(
                {
                    "valid_wmae": compute_wmae(
                        prepared["y_valid"],
                        final_model.predict(final_X_valid),
                    ),
                    "train_rows": len(train_df),
                    "valid_rows": len(prepared["y_valid"]),
                    "test_rows": len(prepared["df_test_processed"]),
                    "feature_count": len(prepared["feature_list"]),
                    "strategy": "full_data_retrain",
                },
                ROOT / "artifacts" / "validation_report.csv",
            )
            logger.info("Final model retrained on all available training data")
            logger.info(f"Model saved      : {model_version_path}")
            model = final_model
    else:
        with model_path.open("rb") as file:
            model = pickle.load(file)
        logger.info(f"Model loaded     : {model_path}")

    if args.mode in {"predict", "full"}:
        # 最終推論では、CVでの設定選定後に再学習した最終モデルを使う。
        if model is None:
            raise ValueError(f"No model is available for final inference strategy='{strategy}'")
        predictions = predict_submission(model, prepared["df_test_processed"], prepared["feature_list"])
        predictions = apply_prediction_correction(predictions, config.get("correction_amount", 0.0))
        valid_wmae = compute_wmae(
            prepared["y_valid"],
            model.predict(validation_X_for_prediction),
        )

        if args.two_stage_snow_prediction:
            winter_flag_column = prepared["df_test_processed"].get("冬季フラグ")
            flag_column = prepared["df_test_processed"].get("着雪量予測フラグ")
            probability_column = prepared["df_test_processed"].get("着雪確率")

            if winter_flag_column is not None:
                predictions = apply_two_stage_snow_prediction(
                    predictions,
                    winter_flag_column,
                    snow_probabilities=probability_column,
                    thresholds=thresholds,
                )
                logger.info(
                    "Two-stage snow prediction enabled: rows with '冬季フラグ'=0 are forced to 0 and "
                    "probabilities below threshold are filtered."
                )
            elif flag_column is not None:
                predictions = apply_two_stage_snow_prediction(predictions, flag_column)
                logger.info("Two-stage snow prediction enabled: rows with '着雪量予測フラグ'=0 are forced to 0.")
            else:
                logger.warning(
                    "Two-stage snow prediction was requested, but neither '冬季フラグ' nor '着雪量予測フラグ' is found in the test dataframe. "
                    "The raw model predictions will be used without gating."
                )

        save_submission(predictions, output_path=args.output_path)
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
