from __future__ import annotations

from pathlib import Path

from numpy.random import RandomState

# fmt: off
import sys  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
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
    Iterator,
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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import column_or_1d
from typing_extensions import Literal

"""key is metric name, tuple[float, ...] is fold values"""
KfoldResult = dict[str, tuple[float, float, float, float, float]]


class NestedKfold:
    """Implements nested k-fold with memoized folds (same inner and outer folds
    for each NestedKfold object)

    Properties
    ----------
    inner_results: KFoldResult
    outer_results: KFoldResult
    """

    pass


class BinStratifiedKFold(StratifiedKFold):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        n_bins: int = 10,
        shuffle: bool = False,
        random_state: int | RandomState | None = None,
    ) -> None:
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)
        self.n_bins = n_bins

    def split(self, X, y, groups: Any = None) -> Iterator[Any]:
        y = np.asarray(y)
        y_type = type_of_target(y)
        allowed = ("binary", "multiclass", "continuous")
        if y_type not in allowed:
            raise ValueError(
                f"Supported target types are: {allowed}. Got {y_type!r} instead."
            )

        if y_type in ("binary", "multiclass"):
            return super().split(X, y, groups)

        # continuous case
        y = column_or_1d(y)
        y_cls = np.full_like(y, -1, dtype=np.int64)
        bin_edges = np.percentile(y, np.linspace(0, 100, self.n_bins + 1, endpoint=True))
        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            idx = (y >= lo) & (y < hi)
            y_cls[idx] = i
        y_cls[y_cls == -1] = i  # handle y <= hi cases
        return super().split(X, y_cls, groups)
