# validationに関する共通関数
import pandas as pd


def split_time(df_train: pd.DataFrame, split_col: str, target_col: str, split_date: pd.Timestamp)-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    時系列でtrainとvalidに分割する関数
    
    Args:
        df_train (pd.DataFrame): 学習用データフレーム
        split_col (str): 分割に使用する列名
        target_col (str): ターゲット変数の列名
        split_date (pd.Timestamp): 分割する日付(YYYY-MM-DD形式)
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 分割されたtrainとvalidのデータフレームとターゲット変数
    """

    # 日付列をdatetime型にする
    df_train[split_col] = pd.to_datetime(df_train[split_col])

    # 時系列で分割
    

    train_mask = df_train[split_col] < split_date
    valid_mask = df_train[split_col] >= split_date

    X_train = df_train.loc[train_mask].drop(columns=[target_col])
    X_valid = df_train.loc[valid_mask].drop(columns=[target_col])

    y_train = df_train.loc[train_mask, target_col]
    y_valid = df_train.loc[valid_mask, target_col]

    # 確認
    print("train:", df_train.loc[train_mask, split_col].min(),
        "～", df_train.loc[train_mask, split_col].max())

    print("valid:", df_train.loc[valid_mask, split_col].min(),
        "～", df_train.loc[valid_mask, split_col].max())

    print("train件数:", len(X_train))
    print("valid件数:", len(X_valid))
    return X_train, X_valid, y_train, y_valid