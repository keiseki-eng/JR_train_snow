"""Repository-root package shim for the project modules.

The actual implementation lives under `30.src/jr_snow`, but Python invoked from the
repository root only sees the repo root by default. This package exposes the same
module namespace so imports like `from jr_snow.logging_utils import setup_logger`
work consistently in notebooks, scripts, and ad-hoc command-line checks.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
_SRC_PACKAGE = _PROJECT_ROOT / "30.src" / "jr_snow"

# Point this package at the real source tree so submodules resolve correctly.
__path__ = [str(_SRC_PACKAGE)] + list(__path__)

# Re-export the public API for a smoother import experience.
from .config import build_feature_columns, load_project_config, load_yaml_config  # noqa: F401,E402
from .cross_validation import time_series_folds  # noqa: F401,E402
from .data import load_path_config, load_train_test_data  # noqa: F401,E402
from .evaluation import compute_wmae, summarize_prediction_stats, summarize_target_stats  # noqa: F401,E402
from .feature_importance import save_feature_importance  # noqa: F401,E402
from .features import prepare_model_inputs  # noqa: F401,E402
from .logging_utils import setup_logger  # noqa: F401,E402
from .model_registry import build_model_version, load_model_artifact, save_model_artifact  # noqa: F401,E402
from .modeling import train_lightgbm_model, wmae_eval  # noqa: F401,E402
from .reporting import save_validation_report  # noqa: F401,E402

__all__ = [
    "build_feature_columns",
    "load_project_config",
    "load_yaml_config",
    "time_series_folds",
    "load_path_config",
    "load_train_test_data",
    "compute_wmae",
    "summarize_prediction_stats",
    "summarize_target_stats",
    "save_feature_importance",
    "prepare_model_inputs",
    "setup_logger",
    "build_model_version",
    "load_model_artifact",
    "save_model_artifact",
    "train_lightgbm_model",
    "wmae_eval",
    "save_validation_report",
]
