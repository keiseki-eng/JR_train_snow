"""時系列データのtrain/valid分割を行う共通ユーティリティ。

学習時に未来データを使わないようにするため、日付で学習区間と検証区間を分ける。
"""
import pandas as pd


def split_time(df_train: pd.DataFrame, split_col: str, target_col: str, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """指定日付より前を学習、以降を検証として分割する。

    Args:
        df_train: 学習用データフレーム
        split_col: 分割に使う日付列
        target_col: 予測対象の列名
        split_date: 分割基準日

    Returns:
        学習用X, 検証用X, 学習用y, 検証用y
    """
    # 日付列を datetime 型に変換して比較可能にする。
    df_train[split_col] = pd.to_datetime(df_train[split_col])

    # 時系列順に、指定日以前を学習データ、指定日以降を検証データに分ける。
    train_mask = df_train[split_col] < split_date
    valid_mask = df_train[split_col] >= split_date

    X_train = df_train.loc[train_mask].drop(columns=[target_col])
    X_valid = df_train.loc[valid_mask].drop(columns=[target_col])

    y_train = df_train.loc[train_mask, target_col]
    y_valid = df_train.loc[valid_mask, target_col]

    # データの範囲と件数を確認し、分割の妥当性を把握しやすくする。
    print("train:", df_train.loc[train_mask, split_col].min(), "～", df_train.loc[train_mask, split_col].max())
    print("valid:", df_train.loc[valid_mask, split_col].min(), "～", df_train.loc[valid_mask, split_col].max())
    print("train件数:", len(X_train))
    print("valid件数:", len(X_valid))
    return X_train, X_valid, y_train, y_valid