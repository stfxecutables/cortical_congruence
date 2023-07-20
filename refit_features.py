from __future__ import annotations

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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import filterwarnings

import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import Booster, CVBooster
from lightgbm import Dataset as LGBMDataset
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import balanced_accuracy_score, explained_variance_score
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing_extensions import Literal

from src.enumerables import FeatureRegex, FreesurferStatsDataset, RegressionModel

from lgbm_select import DEFAULTS

# fmt: off
LANG_PERF_FEAT_IDX = [
    900, 888, 152, 580, 805, 297, 492, 947, 877, 821, 933, 693, 491, 923, 548,
    912, 409, 487, 849, 831, 753, 896, 828, 863, 395, 383, 530, 672, 959, 257,
    273, 978, 861, 779, 357, 785, 477, 158, 346, 552, 313, 884, 419, 879, 841,
    992, 451, 658, 732, 799,
]

INT_G_FEAT_IDX = [
    983, 514, 896, 979, 864, 667, 533, 879, 872, 849, 518, 907, 975, 897, 534,
    625, 594, 350, 614, 890, 413, 560, 909, 270, 459, 996, 829, 503, 951, 353,
    803, 392, 1002, 776, 1001, 595, 833, 4, 297, 210, 621, 373, 383, 576, 446,
    608, 761, 288, 332, 855
]
# fmt: on


@dataclass
class FitArgs:
    params: dict
    is_reg: bool
    X: DataFrame
    y: ndarray


def get_params() -> list[dict]:
    max_depth = 8
    depths = list(range(1, max_depth + 1)) if max_depth > -1 else [-1]
    num_leaves = list(range(2, min(64, int(2**max_depth)))) if max_depth > -1 else [64]
    params = np.array(
        ParameterGrid(
            dict(
                num_leaves=num_leaves,  # less than 2**max_depth
                # max_depth=[-1, 1, 2, 3, 4, 5, 6],
                max_depth=depths,
                min_child_samples=[5, 25, 100, 1000],
                subsample=[0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
                n_estimators=[50, 100, 200],
                reg_alpha=np.logspace(-6, 2, num=16, base=10).tolist(),
                n_jobs=[1],
            )
        )
    )
    params = params[np.random.permutation(len(params))]
    print(f"n_params: {len(params)}")
    return params.tolist()


def fit_lgb(fit_args: FitArgs) -> DataFrame:
    params = fit_args.params
    is_reg = fit_args.is_reg
    X, y = fit_args.X, fit_args.y
    args = {**DEFAULTS, **params}
    estimator = LGBMRegressor(**args) if is_reg else LGBMClassifier(**args)
    scoring = "explained_variance" if is_reg else "balanced_accuracy"
    exp_cvs = cross_val_score(estimator, X, y, scoring=scoring, n_jobs=1)
    mean = np.mean(exp_cvs)
    return DataFrame(
        columns=[*[f"fold{i + 1}" for i in range(5)], "mean"],
        data=[[*exp_cvs, mean]],
        index=[0],
    )


def refit_features(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    selected_idx: list[int],
    n_iter: int = 128,
) -> DataFrame:
    df = dataset.load_complete()
    feats = df.filter(regex=feature_regex.value)
    feature_cols = feats.columns
    df.loc[:, feature_cols] = feats.fillna(feats.mean())
    df.loc[:, feature_cols] = StandardScaler().fit_transform(df[feature_cols])
    target_col = df.filter(regex="TARGET").filter(regex=target_regex).columns
    idx_keep = ~df[target_col].isnull()
    df = df.loc[idx_keep.index]
    is_reg = "REG" in target_col.item()

    params = get_params()
    params = params[:n_iter]
    X = df[feature_cols].iloc[:, selected_idx]
    y = np.ravel(df[target_col])

    args = [FitArgs(params=param, is_reg=is_reg, X=X, y=y) for param in params]

    results: list[DataFrame]
    results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_lgb)(arg) for arg in args)  # type: ignore  # noqa

    result = pd.concat(results, axis=0, ignore_index=True)
    # result.to_parquet(outfile)
    print(result.sort_values(by="mean", ascending=True).round(4).tail(500))
    # print(f"Saved results to {outfile}")
    Xf = np.asfortranarray(X.to_numpy())
    lasso = RegressionModel.Lasso.get()
    scoring = "explained_variance" if is_reg else "balanced_accuracy"
    exp_cvs = cross_val_score(lasso, Xf, y, scoring=scoring, n_jobs=-1)
    print(f"LASSO mean: {np.mean(exp_cvs)}")

    return result


if __name__ == "__main__":
    result = refit_features(
        dataset=FreesurferStatsDataset.HCP,
        feature_regex=FeatureRegex.FS_OR_CMC,
        # target_regex="language_perf",
        # selected_idx=LANG_PERF_FEAT_IDX,
        target_regex="int_g_like",
        selected_idx=slice(None),
        n_iter=1,
    )
