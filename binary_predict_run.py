"""着雪確率予測モデルの実行用スクリプト。

Notebook の `binary_base_line.ipynb` で行っていた処理を、実行可能な Python
モジュールとして整理する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "30.src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jr_snow.config import build_feature_columns, load_project_config
from jr_snow.data import load_train_test_data
from jr_snow.features import build_engineered_feature_frame
from jr_snow.model_registry import save_model_artifact
from utils.validation import split_time


def parse_args() -> argparse.Namespace:
    """CLI 引数を解釈する。"""
    parser = argparse.ArgumentParser(description="Binary snow occurrence prediction run")
    parser.add_argument("--config", type=str, default=str(ROOT / "00.config" / "config.yaml"))
    parser.add_argument("--path-config", type=str, default=str(ROOT / "00.config" / "path.yaml"))
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--model-output-path", type=str, default=str(ROOT / "artifacts" / "binary_model.pkl"))
    parser.add_argument("--prediction-output-path", type=str, default=str(ROOT / "artifacts" / "binary_prediction_probabilities.csv"))
    return parser.parse_args()


def prepare_binary_model_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: dict,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    """Notebook と同じ前処理・分割ロジックを再現する。"""
    feature_list = list(feature_columns.get("feature_list", []))
    categorical_cols = list(feature_columns.get("categorical_cols", []))

    if "冬季フラグ" in df_train.columns:
        df_train = df_train.loc[pd.to_numeric(df_train["冬季フラグ"], errors="coerce") == 1].copy()

    df_train = build_engineered_feature_frame(df_train)
    df_test = build_engineered_feature_frame(df_test)

    fix_list = feature_list + ["年月日", "着雪有無フラグ"]
    df_train = df_train.reindex(columns=fix_list)

    X_train, X_valid, y_train, y_valid = split_time(
        df_train,
        split_col="年月日",
        target_col="着雪有無フラグ",
        split_date=pd.Timestamp(split_date),
    )

    X_train = X_train[feature_list].copy()
    X_valid = X_valid[feature_list].copy()
    df_test_processed = df_test.reindex(columns=feature_list)

    cat_cols = [column for column in categorical_cols if column in X_train.columns]
    obj_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    for frame in [X_train, X_valid, df_test_processed]:
        columns = sorted(set(cat_cols + obj_cols).intersection(frame.columns))
        if columns:
            frame[columns] = frame[columns].astype("category")

    for column in [column for column in X_train.columns if "日照時間" in column]:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_valid[column] = pd.to_numeric(X_valid[column], errors="coerce")
        df_test_processed[column] = pd.to_numeric(df_test_processed[column], errors="coerce")

    return X_train, X_valid, y_train, y_valid, df_test_processed, feature_list, cat_cols


def train_binary_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    categorical_cols: list[str],
    model_params: dict,
) -> lgb.LGBMModel:
    """着雪有無フラグの binary LightGBM を学習する。"""
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_cols)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_cols)

    model = lgb.train(
        model_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[lgb.early_stopping(100)],
    )
    return model


def evaluate_binary_model(model: lgb.LGBMModel, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[float, np.ndarray]:
    """検証データの予測確率と ROC-AUC を計算する。"""
    y_valid_bin = (y_valid > 0).astype(int)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_valid)[:, 1]
    else:
        y_score = model.predict(X_valid).astype(float)

    auc_score = roc_auc_score(y_valid_bin, y_score)
    return float(auc_score), np.asarray(y_score, dtype=float)


def main() -> None:
    """着雪確率予測モデルの実行。"""
    args = parse_args()
    config, path_config = load_project_config(args.config, args.path_config)
    feature_columns = build_feature_columns(config)
    path_map = path_config.get("INTERIUM_PATH", path_config)

    train_df, test_df = load_train_test_data(
        train_data_path=args.train_data or path_map.get("train_data"),
        test_data_path=args.test_data or path_map.get("test_data"),
        path_config=path_config,
    )

    X_train, X_valid, y_train, y_valid, df_test_processed, feature_list, categorical_cols = prepare_binary_model_data(
        df_train=train_df,
        df_test=test_df,
        feature_columns=feature_columns,
        split_date=config["binary_split_date"],
    )

    model = train_binary_model(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        categorical_cols=categorical_cols,
        model_params=config["BINARY_MODEL_PARAMS"],
    )

    auc_score, y_score = evaluate_binary_model(model, X_valid, y_valid)
    print(f"ROC-AUC: {auc_score:.4f}")

    model_path = save_model_artifact(model, ROOT / "artifacts", prefix="binary_model")
    print(f"Model saved: {model_path}")

    probabilities = model.predict(df_test_processed)
    if "冬季フラグ" in test_df.columns:
        winter_flags = pd.to_numeric(test_df["冬季フラグ"], errors="coerce")
        probabilities = np.where(winter_flags.to_numpy() == 1, probabilities, 0.0)
    probability_df = pd.DataFrame({"着雪確率": probabilities})
    output_path = Path(args.prediction_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probability_df.to_csv(output_path, index=False)
    print(f"Binary prediction probabilities saved: {output_path}")

    fpr, tpr, _ = roc_curve((y_valid > 0).astype(int), y_score)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
