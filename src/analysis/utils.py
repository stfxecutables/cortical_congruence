from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from src.feature_selection.kfold import BinStratifiedKFold


def get_kfold(bin_stratify: bool, is_reg: bool) -> BaseCrossValidator:
    if is_reg and bin_stratify:
        return BinStratifiedKFold()
    if is_reg:
        return KFold()
    return StratifiedKFold()
