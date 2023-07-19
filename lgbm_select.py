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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import balanced_accuracy_score, explained_variance_score
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import TABLES
from src.enumerables import FeatureRegex, FreesurferStatsDataset
from src.munging.hcp import PhenotypicFocus

OUT = TABLES / "lgbm_select"


@dataclass
class FitArgs:
    is_reg: bool
    lgb_args: Mapping
    X_train: DataFrame
    X_test: DataFrame
    y_train: DataFrame
    y_test: DataFrame


def fit_lgb(fit_args: FitArgs) -> DataFrame:
    defaults = dict(
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=1.0,
        n_estimators=100,
        reg_alpha=0.0,
        n_jobs=-1,
        verbose=-1,
    )

    param = fit_args.lgb_args
    is_reg = fit_args.is_reg
    X_train = fit_args.X_train
    X_test = fit_args.X_test
    y_train = fit_args.y_train
    y_test = fit_args.y_test

    args = {**defaults, **param}
    estimator = LGBMRegressor(**args) if is_reg else LGBMClassifier(**args)
    scoring = "explained_variance" if is_reg else "balanced_accuracy"
    exp_cv = np.mean(
        cross_val_score(estimator, X_train, y_train, scoring=scoring, n_jobs=1)
    )
    estimator.fit(X_train, y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)
    if is_reg:
        score_train = explained_variance_score(y_train, y_pred_train)
        score_test = explained_variance_score(y_test, y_pred_test)
    else:
        score_train = balanced_accuracy_score(y_train, y_pred_train)
        score_test = balanced_accuracy_score(y_test, y_pred_test)
    cols = (
        ["exp_tr", "exp_cv", "exp_test", "args"]
        if is_reg
        else ["acc_tr", "acc_cv", "acc_test", "args"]
    )
    return DataFrame(
        data=[(score_train, exp_cv, score_test, args)],
        columns=cols,
        index=[0],
    )


def lgbm_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    holdout: float | None = 0.25,
    seed: int | None = None,
) -> DataFrame:
    regex = feature_regex.value
    df = dataset.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )

    target_col = df.filter(regex="TARGET").filter(regex=target_regex).columns
    idx_keep = ~df[target_col].isnull()
    df = df.loc[idx_keep.index]

    is_reg = "REG" in target_col.item()

    X = df.filter(regex=regex)
    y = df[target_col]
    idx_keep = (~y.isna()).values
    X, y = X.iloc[idx_keep], np.ravel(y.iloc[idx_keep])

    if holdout is not None:
        stratify = None if is_reg else np.ravel(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=holdout, stratify=stratify, random_state=seed
        )
    else:
        X_train = X_test = X
        y_train = y_test = y

    max_depth = 4
    depths = list(range(1, max_depth + 1)) if max_depth > -1 else [-1]
    num_leaves = list(range(2, min(64, int(2**max_depth)))) if max_depth > -1 else [64]
    params = np.array(
        ParameterGrid(
            dict(
                num_leaves=num_leaves,  # less than 2**max_depth
                # max_depth=[-1, 1, 2, 3, 4, 5, 6],
                max_depth=depths,
                min_child_samples=[5, 25, 100],
                subsample=[0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
                n_estimators=[100],
                reg_alpha=np.logspace(-6, 2, num=10, base=10).tolist(),
                n_jobs=[1],
            )
        )
    )
    params = params[np.random.permutation(len(params))]
    print(f"n_params: {len(params)}")
    params = params.tolist()[:128]
    args = [
        FitArgs(
            is_reg=is_reg,
            lgb_args=param,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        for param in params
    ]

    results = Parallel(n_jobs=-1, verbose=10)(delayed(fit_lgb)(arg) for arg in args)  # type: ignore  # noqa

    result = pd.concat(results, axis=0, ignore_index=True)
    print(
        result.sort_values(by="exp_cv" if is_reg else "acc_cv", ascending=True)
        .round(4)
        .tail(500)
    )
    return result


def compare_selection(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    holdout: float | None = 0.25,
    n_reps: int = 5,
) -> DataFrame:
    outfile = (
        OUT
        / f"{dataset.value}_{feature_regex}_{target_regex}_n_reps={n_reps}_lgbm_select_seed_compare.parquet"
    )
    if outfile.exists():
        return pd.read_parquet(outfile)

    seeds = np.random.randint(1, int(2**32) - 1, n_reps)
    results = []
    pbar = tqdm(total=int(n_reps**2))
    for seed in seeds:
        for run in range(n_reps):
            result = lgbm_feature_select(
                dataset=dataset,
                feature_regex=feature_regex,
                target_regex=target_regex,
                holdout=holdout,
                seed=int(seed),
            )
            result["seed"] = str(seed)
            result["run"] = run
            results.append(result)
            pbar.update()
    pbar.close()
    df = pd.concat(results, axis=0, ignore_index=True)
    df.to_parquet(outfile)
    print(f"Saved results to {outfile}")
    return df


if __name__ == "__main__":
    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000
    pd.options.display.max_colwidth = 180

    # lgbm_feature_select(
    #     dataset=FreesurferStatsDataset.ABIDE_I,
    #     feature_regex=FeatureRegex.FS,
    #     target_regex="autism",
    #     holdout=0.25,
    # )

    compare_selection(
        dataset=FreesurferStatsDataset.ABIDE_I,
        feature_regex=FeatureRegex.FS,
        # target_regex="int_g_like",
        target_regex="autism",
        holdout=0.25,
        n_reps=5,
    )
    sys.exit()
    df = pd.read_parquet("seed_results.parquet")
    dfs = [d for _, d in df.drop(columns="args").groupby(["seed", "run"])]
    cv_bests = []
    cv_best_10s = []
    for x in dfs:
        cv_bests.append(x.iloc[x.exp_cv.argmax()].to_frame().T.copy())
        idx = x.exp_cv.sort_values(ascending=False).iloc[:10].index
        cv_best_10s.append(x.loc[idx])
    bests = pd.concat(cv_bests, axis=0, ignore_index=True)
    best_10s = pd.concat(cv_best_10s, axis=0, ignore_index=True)

    bests["exp_diff"] = bests["exp_test"] - bests["exp_cv"]
    bests = bests.loc[:, ["seed", "run", "exp_cv", "exp_test", "exp_diff"]]
    print(bests.sort_values(by=["seed", "run", "exp_cv"]).round(3))

    df["exp_diff"] = df["exp_test"] - df["exp_cv"]
    df["overfit"] = df["exp_tr"] - df["exp_cv"]
    # Pearson corr between exp_test and exp_cv only 0.48 (or 0.29 Spearman)

    print(df.corr(numeric_only=True))

    print(
        df.groupby(["seed", "run"])
        .describe()
        .drop(columns=["exp_tr"])
        .drop(columns=["count", "min", "25%", "50%", "75%"], level=1)
        .round(3)
    )
    print(
        df.groupby(["seed"])
        .describe()
        .drop(columns=["exp_tr", "run"])
        .drop(columns=["count", "min", "25%", "50%", "75%"], level=1)
        .round(3)
    )
