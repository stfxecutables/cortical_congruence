from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import ABIDE_I, ABIDE_II
from src.freesurfer import bilateral_stats_to_row, compute_bilateral_stats, merge_stats


def load_row(root: Path) -> DataFrame | None:
    try:
        df = merge_stats(root)
        if df is None:
            return None
        df = compute_bilateral_stats(df)
        df = bilateral_stats_to_row(df)
        return df
    except Exception as e:
        traceback.print_exc()
        print(f"Got error processing stats files at {root}: {e}")
        return None


if __name__ == "__main__":
    abide_i_roots = sorted(ABIDE_I.rglob("stats"))
    abide_ii_roots = sorted(ABIDE_II.rglob("stats"))
    df_abide_is = process_map(
        load_row, abide_i_roots, desc="Loading and parsing ABIDE-I stats"
    )
    df_abide_iis = process_map(
        load_row, abide_ii_roots, desc="Loading and parsing ABIDE-II stats"
    )

    df_abide_i = pd.concat(
        [df for df in df_abide_is if df is not None], axis=0, ignore_index=True
    )
    df_abide_ii = pd.concat(
        [df for df in df_abide_iis if df is not None], axis=0, ignore_index=True
    )

    df = pd.concat([df_abide_i, df_abide_ii], axis=0, ignore_index=True)
    df.to_json(ROOT / "abide_cmc.json")
    df.to_parquet(ROOT / "abide_cmc.parquet")
    df.to_csv(ROOT / "abide_cmc.csv")

    print(df)
    sys.exit()
