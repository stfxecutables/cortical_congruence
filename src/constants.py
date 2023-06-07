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
from typing_extensions import Literal


DATA = ROOT / "data"
ABIDE_I = DATA / "ABIDE-I"
ABIDE_II = DATA / "ABIDE-II"

ABIDE_I_PHENO = DATA / "ABIDE_I_Phenotypic_V1_0b.csv"
ABIDE_II_PHENO = DATA / "ABIDEII_Composite_Phenotypic.csv"


if __name__ == "__main__":
    df = pd.read_csv(ABIDE_I_PHENO)
    df2 = pd.read_csv(ABIDE_II_PHENO, encoding="iso8859_2")
    print(df)
    print(df2)
