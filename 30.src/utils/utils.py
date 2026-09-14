import unicodedata
from pathlib import Path
import pandas as pd

def read_data(path: Path, **kwargs): 
    """
    ファイルの拡張子に応じてpd.DataFrameで読み込む関数

    Args:
        path (Path): 読み込むファイルのパス
        **kwargs: pd.read_csvやpd.read_excelに渡す追加の引数

    Returns:
        pd.DataFrame: 読み込んだデータ
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("dtype", "str")
        return pd.read_csv(path, **kwargs)
    elif suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path, **kwargs)
    elif suffix == ".json":
        return pd.read_json(path, **kwargs)
    elif suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path, **kwargs)
    else:
        raise ValueError(f"対応していないファイル形式です: {suffix}")
    
    
def normalize_text(s: pd.Series) -> pd.Series:
    """
    データフレームのカラムの文字列を正規化する関数
    - 全角数字・英字・記号を半角に変換する
    - 半角・全角の空白を削除
    - ハイフンに相当する記号を半角のハイフンに統一する
    

    Args:
        s (pd.Series): 正規化する文字列のSeries

    Returns:
        pd.Series: 正規化された文字列のSeries
        
    """
    normalized = s.astype("string").map(lambda x: unicodedata.normalize('NFKC', str(x)) if pd.notnull(x) else x)
    return (
        normalized.str.replace(r'\s+', '', regex=True)  # 空白を削除
        .str.replace(r'[-‐‑‒–—―]', '-', regex=True)  # ハイフンに相当する記号を半角のハイフンに統一
    )