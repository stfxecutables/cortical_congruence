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
from tqdm import tqdm
from typing_extensions import Literal

if __name__ == "__main__":
    p = 5000
    n = 50
    feats = np.zeros(p, dtype=bool)
    idx = np.arange(len(feats))
    pbar = tqdm()
    percent = np.mean(feats)
    while percent < 0.99:
        selected = np.random.choice(idx, size=n, replace=False)
        feats[selected] = True
        percent = np.mean(feats)
        pbar.set_description(f"{percent}", refresh=True)
        pbar.update()
    pbar.close()
