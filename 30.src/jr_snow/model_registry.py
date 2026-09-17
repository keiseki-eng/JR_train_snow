"""学習済みモデルの保存と読み込みを行うモジュール。

実験ごとにモデルファイルの識別子を変え、再利用や比較をしやすくする。
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def build_model_version(prefix: str = "lightgbm_model") -> str:
    """モデル名に時刻を付けて、実行ごとの識別子を生成する。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_model_artifact(model: Any, output_dir: str | Path, prefix: str = "lightgbm_model", version: str | None = None) -> Path:
    """学習済みモデルをpickleとして保存する。

    version が未指定のときは実行時刻付きの識別子を自動生成する。
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = build_model_version(prefix)

    model_path = directory / f"{version}.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    return model_path


def load_model_artifact(model_path: str | Path) -> Any:
    """保存済みモデルを読み込む。"""
    with Path(model_path).open("rb") as file:
        return pickle.load(file)
