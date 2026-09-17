from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def build_model_version(prefix: str = "lightgbm_model") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def save_model_artifact(model: Any, output_dir: str | Path, prefix: str = "lightgbm_model", version: str | None = None) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = build_model_version(prefix)

    model_path = directory / f"{version}.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)
    return model_path


def load_model_artifact(model_path: str | Path) -> Any:
    with Path(model_path).open("rb") as file:
        return pickle.load(file)
