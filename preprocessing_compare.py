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
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from typing_extensions import Literal

from src.enumerables import FreesurferStatsDataset


class RobustClipScaler(RobustScaler):
    def __init__(
        self,
        *,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = ...,
        copy: bool = True,
        unit_variance: bool = False,
    ) -> None:
        super().__init__(
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            copy=copy,
            unit_variance=unit_variance,
        )
        self.amin, self.amax = quantile_range

    def fit_transform(self, X: Any, y: Any = None, **fit_params) -> ndarray:
        X = np.copy(X)
        for i in range(X.shape[1]):
            a_min, a_max = np.percentile(X[:, i], [self.amin, self.amax])
            X[:, i] = np.clip(X[:, i], a_min=a_min, a_max=a_max)
        return super().fit_transform(X, y, **fit_params)


class MinMaxClip(MinMaxScaler):
    def __init__(
        self,
        feature_range: tuple[int, int] = ...,
        pmin: float = 5.0,
        pmax: float = 95.0,
        *,
        copy: bool = True,
        clip: bool = False,
    ) -> None:
        super().__init__(feature_range, copy=copy, clip=clip)
        self.pmin, self.pmax = pmin, pmax

    def fit_transform(self, X: Any, y: Any = None, **fit_params) -> ndarray:
        X = np.copy(X)
        for i in range(X.shape[1]):
            a_min, a_max = np.percentile(X[:, i], [self.pmin, self.pmax])
            X[:, i] = np.clip(X[:, i], a_min=a_min, a_max=a_max)

        return super().fit_transform(X, y, **fit_params)


if __name__ == "__main__":
    df = FreesurferStatsDataset.ABIDE_I.load_complete()
    feats = df.filter(regex="FS").copy()
    feats = feats.fillna(feats.mean())
    reg = df.filter(regex="REG").filter(regex="fiq")
    idx_keep = ~reg.iloc[:, 0].isnull()
    feats = feats.loc[idx_keep]
    reg = reg.loc[idx_keep]
    scalers = {
        # "MinMaxScaler": lambda: MinMaxScaler(),
        "StandardScaler": lambda: StandardScaler(),
        # "QuantileNormal": lambda: QuantileTransformer(output_distribution="normal"),
        # "QuantileUniform": lambda: QuantileTransformer(output_distribution="uniform"),
        # "RobustScaler": lambda: RobustScaler(quantile_range=(10.0, 90.0)),
        # "RobustScaler75": lambda: RobustScaler(quantile_range=(25.0, 75.0)),
        # "RobustClip95": lambda: RobustClipScaler(quantile_range=(5.0, 95.0)),
        # "RobustClip90": lambda: RobustClipScaler(quantile_range=(10.0, 90.0)),
        # "RobustClip75": lambda: RobustClipScaler(quantile_range=(25.0, 75.0)),
        # "MinMaxClip98": lambda: MinMaxClip((0, 1), pmin=2.0, pmax=98.0),
        # "MinMaxClip95": lambda: MinMaxClip((0, 1), pmin=5.0, pmax=95.0),
        # "MinMaxClip90": lambda: MinMaxClip((0, 1), pmin=10.0, pmax=90.0),
        # "MinMaxClip75": lambda: MinMaxClip((0, 1), pmin=25.0, pmax=75.0),
    }
    messages = []
    for scaler_name, constructor in scalers.items():
        scaler = constructor()

        X = np.asfortranarray(scaler.fit_transform(feats.to_numpy()))
        y = np.asfortranarray(reg.to_numpy().ravel())
        # alphas = [1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        # alphas = np.logspace(start=-2, stop=2, num=50, base=10)  # 0.019
        alphas = np.logspace(start=-2, stop=5, num=50, base=10)  # better
        # alphas = np.logspace(start=-2, stop=5, num=100, base=10)  # not really better
        # alphas = np.logspace(start=-2, stop=5, num=200, base=10)  # also not better
        lasso = LassoCV(
            precompute=True,
            selection="random",
            max_iter=10000,
            n_jobs=4,
            alphas=alphas,
        )
        with catch_warnings(record=False):
            filterwarnings("once", category=ConvergenceWarning)
            results = cross_validate(
                lasso, X, y, cv=4, n_jobs=4, scoring="explained_variance"
            )
        scores = results["test_score"].tolist()
        mean = np.round(np.mean(scores), 4)
        scores = "  ".join([f"{round(score, 5): 0.5f}" for score in scores])
        message = f"{scaler_name:<20}: {scores} ({mean})"
        messages.append(message)
        print(message)

    for message in messages:
        print(message)

    # MinMaxScaler        : [ 0.      0.     -0.     -0.0631] (-0.0158)
    # StandardScaler      : [ 0.0194  0.0195 -0.0098  0.0161] (0.0113)
    # QuantileNormal      : [ 0.0125  0.0217  0.0089 -0.0367] (0.0016)
    # QuantileUniform     : [ 0.  0. -0.  0.] (-0.0)
    # RobustScaler90      : [-0.0001 -0.005  -0.0158  0.0073] (-0.0034)
    # RobustScaler75      : ['0.00046', '-0.00205', '-0.06088', '0.00176'] (-0.0152)
    # RobustClip95        : [ 0.  0. -0.  0.] (-0.0)
    # RobustClip90        : [0.e+00 0.e+00 9.e-05 0.e+00] (0.0)
    # RobustClip75        : [0.00561 0.00306 0.00175 0.00504] (0.0039)
    # MinMaxClip90        : ['0.00000', '0.00000', '0.00006', '0.00000'] (0.0)
    # MinMaxClip75        : ['0.00557', '0.00322', '0.00125', '0.00503'] (0.0038)
    # MinMaxClip98        : ['0.00000', '0.00000', '-0.00000', '0.00000'] (-0.0)
    # MinMaxClip95        : ['0.00000', '0.00000', '-0.00000', '0.00000'] (-0.0)
    # MinMaxClip90        : ['0.00000', '0.00000', '0.00006', '0.00000'] (0.0)
    # MinMaxClip75        : ['0.00557', '0.00322', '0.00125', '0.00503'] (0.0038)

    # With max_iter = 10000
    # StandardScaler      : 0.01958  0.01952  -0.00984  0.01607 (0.0113)
    # QuantileNormal      : 0.01248  0.02167  0.00892  -0.03669 (0.0016)
    # RobustScaler75      : 0.00046  -0.00206  -0.06096  0.00176 (-0.0152)
    # MinMaxClip75        : 0.00557  0.00322  0.00125  0.00503 (0.0038)
    print()
