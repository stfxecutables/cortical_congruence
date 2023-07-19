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
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from typing_extensions import Literal

from src.enumerables import ClassificationMetric, FreesurferStatsDataset

if __name__ == "__main__":
    strategies = ["prior", "stratified", "uniform"]
    scorers = ClassificationMetric.scorers()

    all_results = []
    for ds in [
        FreesurferStatsDataset.ABIDE_I,
        FreesurferStatsDataset.ABIDE_II,
        FreesurferStatsDataset.ADHD_200,
        FreesurferStatsDataset.HCP,
    ]:
        df = ds.load_complete()
        cols = (
            df.filter(regex="CLS").columns.to_list()
            + df.filter(regex="MULTI").columns.to_list()
        )
        for target_col in cols:
            y = df[target_col].dropna()
            for strat in strategies:
                model = DummyClassifier(strategy=strat)

                results = cross_validate(
                    model,
                    y,
                    y,
                    cv=5,
                    scoring=scorers,
                    n_jobs=5,
                    error_score="raise",
                )
                result = DataFrame(results)
                result.insert(0, "data", ds.value)
                result.insert(0, "target", str(target_col).replace("TARGET__", ""))
                result.insert(0, "strat", strat)
                result["N"] = len(y)
                all_results.append(
                    result.rename(columns=lambda s: s.replace("test_", "")).drop(
                        columns=["fit_time", "score_time"]
                    )
                )
    df = pd.concat(all_results, axis=0, ignore_index=True)
    df.target = df.target.str.replace("CLS__", "")
    pd.options.display.max_rows = 500
    print(df.groupby(["data", "target", "strat"]).max().round(3))
    print(df.drop(columns="strat").groupby(["data", "target"]).max().round(3))
