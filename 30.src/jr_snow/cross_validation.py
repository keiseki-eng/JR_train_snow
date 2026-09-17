"""時系列データに対する交差検証用のFoldを作るモジュール。

過去の期間を学習に使い、将来の期間を検証に使う構造で、時間順の情報漏れを防ぐ。
"""
from __future__ import annotations
import logging
import pandas as pd
log = logging.getLogger(__name__)


def time_series_folds(
    df: pd.DataFrame,
    split_col: str = "年月日",
    target_col: str = "合計",
    n_splits: int = 3,
    min_train_size: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """時系列CV用のfoldを生成する。

    expanding-window形式。
    各foldでは、検証期間より前のデータを学習に使用する。
    """

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    working_df = df.copy()
    working_df[split_col] = pd.to_datetime(working_df[split_col])
    working_df = working_df.sort_values(split_col).reset_index(drop=True)

    # ==========================================================
    # 基本情報
    # ==========================================================
    log.info("===== Time Series CV Start =====")
    log.info(f"Total rows       : {len(working_df)}")
    log.info(f"Total unique dates: {working_df[split_col].nunique()}")
    log.info(f"Target           : {target_col}")
    log.info(f"Target mean      : {working_df[target_col].mean():.8f}")
    log.info(f"Target median    : {working_df[target_col].median():.8f}")
    log.info(f"Target max       : {working_df[target_col].max():.8f}")
    log.info(
        f"Target > 0      : "
        f"{(working_df[target_col] > 0).sum()} "
        f"/ {len(working_df)} "
        f"({(working_df[target_col] > 0).mean():.4%})"
    )

    unique_dates = sorted(
        working_df[split_col].dropna().unique()
    )

    if len(unique_dates) < n_splits + 1:
        raise ValueError(
            "Not enough unique dates for the requested number of folds."
        )

    # expanding-window CV
    boundaries = [
        int(round(len(unique_dates) * (idx + 1) / (n_splits + 1)))
        for idx in range(n_splits)
    ]

    log.info(f"CV fold count    : {n_splits}")
    log.info(f"CV boundaries    : {boundaries}")

    folds = []

    for fold_idx, valid_start in enumerate(boundaries):

        valid_end = (
            len(unique_dates)
            if fold_idx == n_splits - 1
            else boundaries[fold_idx + 1]
        )

        if valid_end <= valid_start:
            log.warning(
                f"Fold {fold_idx + 1}: "
                f"invalid date range "
                f"({valid_start} -> {valid_end})"
            )
            continue

        valid_dates = unique_dates[valid_start:valid_end]

        valid_mask = working_df[split_col].isin(valid_dates)
        train_mask = working_df[split_col] < valid_dates[0]

        if train_mask.sum() < min_train_size:
            log.warning(
                f"Fold {fold_idx + 1}: "
                f"train size {train_mask.sum()} "
                f"< min_train_size {min_train_size}"
            )
            continue

        if valid_mask.sum() == 0:
            log.warning(
                f"Fold {fold_idx + 1}: validation data is empty"
            )
            continue

        X_train = working_df.loc[train_mask].drop(columns=[target_col])
        X_valid = working_df.loc[valid_mask].drop(columns=[target_col])

        y_train = working_df.loc[train_mask, target_col]
        y_valid = working_df.loc[valid_mask, target_col]

        # ======================================================
        # Fold診断ログ
        # ======================================================
        log.info(f"===== Fold {fold_idx + 1} =====")

        log.info(
            f"Train period     : "
            f"{working_df.loc[train_mask, split_col].min()} "
            f"-> "
            f"{working_df.loc[train_mask, split_col].max()}"
        )

        log.info(
            f"Valid period     : "
            f"{working_df.loc[valid_mask, split_col].min()} "
            f"-> "
            f"{working_df.loc[valid_mask, split_col].max()}"
        )

        log.info(
            f"Train rows       : {len(y_train)}"
        )

        log.info(
            f"Valid rows       : {len(y_valid)}"
        )

        # Target統計
        log.info(
            f"Train target     : "
            f"mean={y_train.mean():.8f}, "
            f"median={y_train.median():.8f}, "
            f"min={y_train.min():.8f}, "
            f"max={y_train.max():.8f}"
        )

        log.info(
            f"Valid target     : "
            f"mean={y_valid.mean():.8f}, "
            f"median={y_valid.median():.8f}, "
            f"min={y_valid.min():.8f}, "
            f"max={y_valid.max():.8f}"
        )

        # 着雪あり件数
        train_positive = (y_train > 0).sum()
        valid_positive = (y_valid > 0).sum()

        log.info(
            f"Train target > 0: "
            f"{train_positive}/{len(y_train)} "
            f"({train_positive / len(y_train):.4%})"
        )

        log.info(
            f"Valid target > 0: "
            f"{valid_positive}/{len(y_valid)} "
            f"({valid_positive / len(y_valid):.4%})"
        )

        # Train / Valid の期間重複チェック
        train_dates = set(
            working_df.loc[train_mask, split_col].dropna().unique()
        )
        valid_dates_set = set(valid_dates)

        overlap_dates = train_dates & valid_dates_set

        log.info(
            f"Date overlap     : {len(overlap_dates)}"
        )

        if overlap_dates:
            log.error(
                f"DATE LEAKAGE DETECTED: "
                f"{len(overlap_dates)} overlapping dates"
            )

        # 時系列順序チェック
        train_max_date = working_df.loc[
            train_mask, split_col
        ].max()

        valid_min_date = working_df.loc[
            valid_mask, split_col
        ].min()

        log.info(
            f"Train max date   : {train_max_date}"
        )

        log.info(
            f"Valid min date   : {valid_min_date}"
        )

        if train_max_date >= valid_min_date:
            log.error(
                "TIME ORDER ERROR: "
                "train max date >= valid min date"
            )

        folds.append(
            (X_train, X_valid, y_train, y_valid)
        )

    log.info(
        f"===== Time Series CV End: "
        f"{len(folds)} folds generated ====="
    )

    return folds