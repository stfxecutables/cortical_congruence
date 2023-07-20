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

PERCENTILES = (
    np.array([0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.5, 99.0, 99.5])
    / 100
)
RS = np.arange(start=0.0, stop=0.2, step=0.01)
FEATURE_MAX = 5


def count_corrs(
    dist: str,
    n_uninformative: int = 5000,
    n_informative: int = 500,
) -> DataFrame:
    n = 1000
    p = n_uninformative
    m = n_informative
    weights = np.random.uniform(0, 5, [m, 1])
    informatives = np.empty([n, m])
    noise_maxes = -np.sort(-np.linspace(0.0, 10, 11))
    noises = np.empty([len(noise_maxes), n, m])
    for i in range(m):
        fmax = float(np.random.uniform(0, FEATURE_MAX, [1]))
        informatives[:, i] = np.random.uniform(0, fmax, [n])
        for k in range(len(noise_maxes)):
            noise_max = noise_maxes[k]
            noise = np.random.normal(0, noise_max, [n])
            noises[k, :, i] = noise

    if dist in ["norm", "normal"]:
        uninformative = DataFrame(np.random.standard_normal([n, p]))
    elif dist in ["skew", "skewed"]:
        uninformative = DataFrame(np.random.standard_gamma(3.0, [n, p]))
    elif dist in ["exp", "exponential"]:
        uninformative = DataFrame(np.random.standard_exponential([n, p]))
    else:
        raise ValueError()
    dfs = []

    # need:
    # y = sum (w_i * x_i + e_i)
    #   = sum (w_i * x_i) + sum(e_i)
    for k in range(len(noises)):
        target = Series(
            ((informatives @ weights).ravel() + np.sum(noises[k], axis=1)) / m
        )
        inf_df = DataFrame(informatives)

        inf_corrs = inf_df.corrwith(target).abs()
        inf_cnts = []
        for r in RS:
            inf_cnts.append(np.sum(inf_corrs >= r))
        nmax = np.round(noise_maxes[k] / FEATURE_MAX * 100, 0)
        dfs.append(
            DataFrame(
                {f"noise={nmax}%": inf_cnts},
                index=Series(data=RS, name="r >="),
            )
        )

    cnts = []
    uninf_corrs = uninformative.corrwith(target).abs()
    for r in RS:
        cnts.append(np.sum(uninf_corrs >= r))
    rands = Series(name=dist, data=cnts, index=Series(data=RS, name="r >="))
    return pd.concat([rands, *dfs], axis=1)


if __name__ == "__main__":
    dfs = []
    for dist in ["norm", "skew", "exp"]:
        print(count_corrs(dist))
        break
    sys.exit()
    df = (
        pd.concat(dfs, axis=1)
        .drop(columns=["actual-norm", "actual-skew"])
        .rename(columns={"actual-exp": "actual"})
    )
    print(df)
