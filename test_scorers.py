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
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_validate
from typing_extensions import Literal

from src.enumerables import RegressionMetric

if __name__ == "__main__":
    x = np.random.uniform(0, 1, 100).reshape(-1, 1)
    e = np.random.standard_normal(100)
    y = 2 * x + e

    scorers = {reg.value: reg.scorer() for reg in RegressionMetric}
    results = cross_validate(LR(), x, y, scoring=scorers)
    print(results)
