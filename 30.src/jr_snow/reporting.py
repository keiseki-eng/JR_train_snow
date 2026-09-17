from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_validation_report(metrics: dict[str, Any], output_path: str | Path) -> pd.DataFrame:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame([metrics])
    report_df.to_csv(output, index=False)
    return report_df
