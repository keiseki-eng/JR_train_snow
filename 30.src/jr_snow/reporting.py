"""検証結果のCSV保存を行うモジュール。

学習結果の簡易レポートを残して、後から比較できるようにするためのコード。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_validation_report(metrics: dict[str, Any], output_path: str | Path) -> pd.DataFrame:
    """検証メトリクスを1行のCSVとして保存する。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame([metrics])
    report_df.to_csv(output, index=False)
    return report_df
