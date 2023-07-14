from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from pandas import DataFrame


def remove_nans(complete: DataFrame) -> DataFrame:
    # uncontroversial
    df = complete.dropna(axis="columns", how="all")
    df = df.dropna(axis="rows", how="all")  # type: ignore
    # NOTE: Need to really limit to the inner join, which gives number of
    nonnull_counts = len(df) - df.isnull().sum()
    # keep features where more than half of subjects not missing data
    keep = nonnull_counts > 0.5 * len(df)
    df = df.loc[:, keep].copy()
    return df
