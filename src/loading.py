from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
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
from sklearn.preprocessing import StandardScaler
from typing_extensions import Literal

from src.constants import CMC_TABLE


def load_abide_data(
    abide: Literal["I", "II", "both"] = "I",
    regex: str = "CMC",
    target: Literal["autism", "age", "sex", "fiq", "viq", "piq"] = "autism",
    standardize: bool = False,
) -> tuple[DataFrame, Series]:
    df = pd.read_parquet(CMC_TABLE)
    if abide != "both":
        df = df.loc[df["abide"] == abide]
    df = pd.concat([df[target], df.filter(regex=regex)], axis=1)
    cmc = df.filter(regex=regex)
    y = df[target]
    keep_idx = df.isnull().sum() == 0
    cmc = cmc.loc[:, keep_idx]
    keep_idx = ~y.isnull()
    cmc = cmc.loc[keep_idx]
    y = y.loc[keep_idx]
    x = (
        DataFrame(StandardScaler().fit_transform(cmc), columns=cmc.columns)
        if standardize
        else cmc
    )
    return x, y
