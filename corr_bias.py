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


def noise_from_percent(percent: ndarray) -> ndarray:
    return percent / (1 - percent)


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
    # must not include endpoint or get divide by zero error
    noise_maxes = noise_from_percent(
        -np.sort(-np.array(np.linspace(0.0, 1.0, 20, endpoint=False).tolist()))
    )
    fmaxes = []
    noises = np.empty([len(noise_maxes), n, m])
    for i in range(m):
        fmax = float(np.random.uniform(0, FEATURE_MAX, [1]))
        fmaxes.append(fmax)
        informatives[:, i] = np.random.uniform(0, fmax, [n])
        for k in range(len(noise_maxes)):
            noise_max = noise_maxes[k]
            noise = np.random.uniform(0, noise_max * fmax, [n])
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
        # if noise in Unif(-M, M), data has range from -rM to rM, and final is
        # data + noise, then noise is (M / (rM + M)) = 1 / (r*M) 50%.
        nmax = noise_maxes[k] / (1 + noise_maxes[k])
        noise_percent = np.round(nmax * 100, 0)
        dfs.append(
            DataFrame(
                {f"{noise_percent}": inf_cnts},
                index=Series(data=RS, name="|r| >="),
            )
        )

    cnts = []
    uninf_corrs = uninformative.corrwith(target).abs()
    for r in RS:
        cnts.append(np.sum(uninf_corrs >= r))
    rands = Series(name=dist, data=cnts, index=Series(data=RS, name="|r| >="))
    results = pd.concat([rands, *dfs], axis=1)
    results.columns.name = "noise (%)"
    return results


if __name__ == "__main__":
    dfs = []
    for dist in ["norm", "skew"]:
        print(count_corrs(dist, n_informative=50))
        print(count_corrs(dist, n_informative=500))
    sys.exit()
    df = (
        pd.concat(dfs, axis=1)
        .drop(columns=["actual-norm", "actual-skew"])
        .rename(columns={"actual-exp": "actual"})
    )
    print(df)
