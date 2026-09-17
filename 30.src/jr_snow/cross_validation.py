"""時系列データに対する交差検証用のFoldを作るモジュール。

過去の期間を学習に使い、将来の期間を検証に使う構造で、時間順の情報漏れを防ぐ。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def time_series_folds(
    df: pd.DataFrame,
    split_col: str = "年月日",
    target_col: str = "合計",
    n_splits: int = 3,
    min_train_size: int = 1,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """時系列CV用のfoldを生成する。

    ここでは "expanding-window" 形式を採用する。
    すなわち各 fold では、検証期間の直前までのデータを学習に使い、
    その後ろにある将来の期間を検証に使う。これにより、train 期間は
    fold が進むにつれて広がっていく構造になる。
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    working_df = df.copy()
    working_df[split_col] = pd.to_datetime(working_df[split_col])
    working_df = working_df.sort_values(split_col).reset_index(drop=True)

    unique_dates = sorted(working_df[split_col].dropna().unique())
    if len(unique_dates) < n_splits + 1:
        raise ValueError("Not enough unique dates for the requested number of folds.")

    # expanding-window CV を実現するため、検証区間をデータの後半から順に配置し、
    # それぞれの検証開始位置より前の時点までを学習データとする。
    boundaries = [
        int(round(len(unique_dates) * (idx + 1) / (n_splits + 1)))
        for idx in range(n_splits)
    ]

    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = []
    for fold_idx, valid_start in enumerate(boundaries):
        valid_end = len(unique_dates) if fold_idx == n_splits - 1 else boundaries[fold_idx + 1]
        if valid_end <= valid_start:
            continue

        valid_dates = unique_dates[valid_start:valid_end]
        valid_mask = working_df[split_col].isin(valid_dates)
        train_mask = working_df[split_col] < valid_dates[0]

        if train_mask.sum() < min_train_size or valid_mask.sum() == 0:
            continue
        X_train = working_df.loc[train_mask].drop(columns=[target_col])
        X_valid = working_df.loc[valid_mask].drop(columns=[target_col])
        y_train = working_df.loc[train_mask, target_col]
        y_valid = working_df.loc[valid_mask, target_col]
        folds.append((X_train, X_valid, y_train, y_valid))

    return folds
