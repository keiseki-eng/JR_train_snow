from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_path_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        from .config import CONFIG_DIR

        path = CONFIG_DIR / "path.yaml"
    if isinstance(path, str):
        path = Path(path)

    from .config import load_yaml_config

    return load_yaml_config(path).get("INTERIUM_PATH", {})


def load_train_test_data(
    train_data_path: str | Path | None = None,
    test_data_path: str | Path | None = None,
    path_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path_config is None:
        path_config = load_path_config()

    path_map = path_config.get("INTERIUM_PATH", path_config)

    train_path = (
        Path(train_data_path)
        if train_data_path is not None
        else Path(path_map["train_data"])
    )
    test_path = (
        Path(test_data_path)
        if test_data_path is not None
        else Path(path_map["test_data"])
    )

    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)
    return train_df, test_df
