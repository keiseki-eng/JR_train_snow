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
    """時系列CV用のfoldを作成する。"""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    working_df = df.copy()
    working_df[split_col] = pd.to_datetime(working_df[split_col])
    working_df = working_df.sort_values(split_col).reset_index(drop=True)

    unique_dates = sorted(working_df[split_col].dropna().unique())
    if len(unique_dates) < n_splits + 1:
        raise ValueError("Not enough unique dates for the requested number of folds.")

    fold_dates = []
    chunk_size = max(1, len(unique_dates) // n_splits)
    for i in range(n_splits):
        start = i * chunk_size
        end = len(unique_dates) if i == n_splits - 1 else (i + 1) * chunk_size
        if end <= start:
            continue
        fold_dates.append(unique_dates[start:end])

    folds: list[tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = []
    for valid_dates in fold_dates:
        valid_mask = working_df[split_col].isin(valid_dates)
        train_mask = ~valid_mask
        if train_mask.sum() < min_train_size or valid_mask.sum() == 0:
            continue
        X_train = working_df.loc[train_mask].drop(columns=[target_col])
        X_valid = working_df.loc[valid_mask].drop(columns=[target_col])
        y_train = working_df.loc[train_mask, target_col]
        y_valid = working_df.loc[valid_mask, target_col]
        folds.append((X_train, X_valid, y_train, y_valid))

    return folds
