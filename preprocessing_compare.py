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
from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import (
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import REGULARIZATION_ALPHAS
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


def test_lasso() -> None:
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
    for scaler_name, constructor in tqdm(scalers.items(), leave=True):
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


def test_lr_cv() -> None:
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::ConvergenceWarning"
    df = FreesurferStatsDataset.ABIDE_I.load_complete()
    feats = df.filter(regex="FS").copy()
    feats = feats.fillna(feats.mean())
    cls = df.filter(regex="CLS").filter(regex="autism")
    idx_keep = ~cls.iloc[:, 0].isnull()
    feats = feats.loc[idx_keep]
    cls = cls.loc[idx_keep]
    scalers = {
        "MinMaxScaler": lambda: MinMaxScaler(),  # consistently better
        # "StandardScaler": lambda: StandardScaler(),
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
    rows = []
    for scaler_name, constructor in tqdm(scalers.items(), leave=True):
        scaler = constructor()

        for _ in tqdm(range(50), leave=False):
            for max_iter in [500]:
                n_feat = int(np.random.randint(2, 51))
                idx = np.random.permutation(feats.shape[1])[:n_feat]
                X = np.asfortranarray(scaler.fit_transform(feats.iloc[:, idx].to_numpy()))
                y = np.asfortranarray(cls.to_numpy().ravel())
                # alphas = [1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
                # alphas = np.logspace(start=-2, stop=2, num=50, base=10)  # 0.019
                alphas = np.logspace(start=-2, stop=5, num=50, base=10)  # better
                # alphas = np.logspace(start=-2, stop=5, num=100, base=10)  # not really better
                # alphas = np.logspace(start=-2, stop=5, num=200, base=10)  # also not better
                lasso = LogisticRegressionCV(
                    Cs=[1 / (2 * alpha) for alpha in alphas],
                    cv=3,
                    max_iter=max_iter,
                    # penalty="l1",
                    penalty="l1",
                    solver="saga",
                    n_jobs=4,
                )
                results = cross_validate(
                    lasso, X, y, cv=4, n_jobs=4, scoring="balanced_accuracy"
                )
                scores = results["test_score"].tolist()
                row = DataFrame(
                    data=[
                        [scaler_name, max_iter] + scores + [np.mean(scores)] + [n_feat]
                    ],
                    columns=[
                        "scaler",
                        "max_iter",
                        "fold1",
                        "fold2",
                        "fold3",
                        "fold4",
                        "mean",
                        "n_feat",
                    ],
                    index=[0],
                )
                mean = np.round(np.mean(scores), 4)
                scores = "  ".join([f"{round(score, 5): 0.5f}" for score in scores])
                message = f"{scaler_name:<20}: {scores} ({mean}) [n_feats={n_feat:2d}]"
                messages.append(message)
                rows.append(row)

    summary = pd.concat(rows, axis=0, ignore_index=True)
    with pd.option_context("display.max_rows", 1000):
        print(summary.sort_values(by="mean", ascending=True))
    # for message in messages:
    #     print(message)

    # MinMaxScaler        :  0.56748   0.52249   0.55861   0.55887 (0.5519)
    # StandardScaler      :  0.56085   0.54582   0.55605   0.55746 (0.555)
    # StandardScaler      :  0.54270   0.45603   0.54697   0.56492 (0.5277)
    # QuantileNormal      :  0.52152   0.50689   0.54371   0.58287 (0.5387)

    #          scaler  max_iter     fold1     fold2     fold3     fold4      mean  n_feat
    #  StandardScaler        50  0.524544  0.546029  0.558840  0.548168  0.544395      44
    # QuantileUniform        50  0.496140  0.574470  0.581437  0.527372  0.544855      18
    # QuantileUniform        50  0.565832  0.582846  0.533426  0.499426  0.545382      31
    # QuantileUniform       100  0.536881  0.510594  0.582846  0.554431  0.546188       6
    #  StandardScaler       500  0.571961  0.480508  0.536713  0.595893  0.546269      22
    #    MinMaxScaler       500  0.562493  0.555996  0.527633  0.539479  0.546400       5
    # QuantileUniform       500  0.566745  0.534156  0.520170  0.569591  0.547666      50
    #  QuantileNormal        50  0.547679  0.539009  0.565833  0.539036  0.547889       2
    #  StandardScaler       500  0.551017  0.561632  0.540445  0.542767  0.548965      27
    #    MinMaxScaler        50  0.538393  0.514090  0.553439  0.594275  0.550049      44
    #    MinMaxScaler       500  0.567658  0.563746  0.526433  0.545115  0.550738      47
    # QuantileUniform       500  0.530621  0.578436  0.538592  0.555840  0.550872      20
    # QuantileUniform       500  0.530986  0.546733  0.585429  0.548586  0.552934      18
    #    MinMaxScaler       100  0.586776  0.566773  0.550464  0.515708  0.554930       8
    #  QuantileNormal        50  0.555634  0.588926  0.510594  0.570765  0.556480       9
    #  StandardScaler       100  0.618414  0.552996  0.512890  0.546968  0.557817      32
    # QuantileUniform       100  0.566562  0.543680  0.542741  0.582898  0.558970      33
    #    MinMaxScaler       100  0.587454  0.542741  0.533921  0.572148  0.559066      42
    #    MinMaxScaler       500  0.602634  0.567712  0.574001  0.571443  0.578948      44


def test_ridge_cv() -> None:
    df = FreesurferStatsDataset.ABIDE_I.load_complete()
    feats = df.filter(regex="FS").copy()
    feats = feats.fillna(feats.mean())
    cls = df.filter(regex="CLS").filter(regex="autism")
    idx_keep = ~cls.iloc[:, 0].isnull()
    feats = feats.loc[idx_keep]
    cls = cls.loc[idx_keep]
    scalers = {
        # "MinMaxScaler": lambda: MinMaxScaler(),  # consistently better
        "StandardScaler": lambda: StandardScaler(),
    }
    rows = []
    for scaler_name, constructor in tqdm(scalers.items(), leave=True):
        scaler = constructor()

        for _ in tqdm(range(50), leave=False):
            for max_iter in [500]:
                n_feat = int(np.random.randint(2, 51))
                idx = np.random.permutation(feats.shape[1])[:n_feat]
                X = np.asfortranarray(scaler.fit_transform(feats.iloc[:, idx].to_numpy()))
                y = np.asfortranarray(cls.to_numpy().ravel())
                # alphas = [1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
                # alphas = np.logspace(start=-2, stop=2, num=50, base=10)  # 0.019
                alphas = np.logspace(start=-2, stop=5, num=50, base=10)  # better
                # alphas = np.logspace(start=-2, stop=5, num=100, base=10)  # not really better
                alphas = np.logspace(
                    start=-5, stop=5, num=200, base=10
                )  # also not better
                lasso = RidgeClassifierCV(
                    alphas=alphas,
                    cv=3,
                )
                results = cross_validate(
                    lasso, X, y, cv=4, n_jobs=4, scoring="balanced_accuracy"
                )
                scores = results["test_score"].tolist()
                row = DataFrame(
                    data=[
                        [scaler_name, max_iter] + scores + [np.mean(scores)] + [n_feat]
                    ],
                    columns=[
                        "scaler",
                        "max_iter",
                        "fold1",
                        "fold2",
                        "fold3",
                        "fold4",
                        "mean",
                        "n_feat",
                    ],
                    index=[0],
                )
                rows.append(row)

    summary = pd.concat(rows, axis=0, ignore_index=True)
    with pd.option_context("display.max_rows", 1000):
        print(summary.sort_values(by="mean", ascending=True))


if __name__ == "__main__":
    # test_lr_cv()
    test_ridge_cv()
