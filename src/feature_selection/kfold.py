from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
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
