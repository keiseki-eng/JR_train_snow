"""ログの出力形式とファイル名管理を担当するモジュール。

実行ごとに別ファイルを作成したい場合や、標準出力とファイル出力を両立したい場合に使う。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def _build_timestamped_log_path(log_path: str | Path) -> Path:
    """ファイル名に実行時刻を付けたログパスを作る。"""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if path.suffix:
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    return path.with_name(f"{path.name}_{timestamp}.log")


def setup_logger(log_path: str | Path | None = None) -> logging.Logger:
    """標準出力とファイル出力を両方持つロガーを初期化する。

    一度の実行ごとに異なるログファイルを作るため、古い pipeline.log を削除して
    pipeline_YYYYMMDD_HHMMSS.log のような新しいファイルに出力する。
    """
    logger = logging.getLogger("jr_snow")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        legacy_log_path = log_path
        if legacy_log_path.exists() and legacy_log_path.is_file() and "_" not in legacy_log_path.stem:
            legacy_log_path.unlink()

        log_file = _build_timestamped_log_path(log_path)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
