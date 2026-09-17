"""Project-local Python import setup.

When this repository is launched from the project root, Python adds the repo root to
sys.path but not the 30.src directory where the project modules live. This file is
imported automatically by Python when present on sys.path, and it ensures that the
source tree is importable as `jr_snow`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "30.src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
