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
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import TABLES, ensure_dir
from src.enumerables import FeatureRegex, FreesurferStatsDataset
from src.munging.hcp import PhenotypicFocus

OUT = ensure_dir(TABLES / "lgbm_select")


@dataclass
class FitArgs:
    is_reg: bool
    lgb_args: Mapping
    X_train: DataFrame
    X_test: DataFrame
    y_train: ndarray
    y_test: ndarray


@dataclass
class LgbmEvalArgs:
    base_params: dict
    params: dict
    data_train: LGBMDataset
    is_reg: bool
    X_train: DataFrame
    X_test: DataFrame
    y_train: ndarray
    y_test: ndarray


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


def data_setup(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    holdout: float | None = 0.25,
    seed: int | None = None,
) -> tuple[bool, bool, DataFrame, DataFrame, ndarray, ndarray]:
    regex = feature_regex.value
    df = dataset.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )

    target_col = df.filter(regex="TARGET").filter(regex=target_regex).columns
    idx_keep = ~df[target_col].isnull()
    df = df.loc[idx_keep.index]

    is_reg = "REG" in target_col.item()
    is_bin = "CLS" in target_col.item()

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
    return is_reg, is_bin, X_train, X_test, y_train, y_test


def get_params() -> list[dict]:
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
    return params.tolist()


def lgbm_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    holdout: float | None = 0.25,
    seed: int | None = None,
) -> DataFrame:
    is_reg, is_bin, X_train, X_test, y_train, y_test = data_setup(
        dataset=dataset,
        feature_regex=feature_regex,
        target_regex=target_regex,
        holdout=holdout,
        seed=seed,
    )

    params = get_params()
    params = params[:128]
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

    results: list[DataFrame]
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
    outfile = OUT / (
        f"{dataset.value}_{feature_regex.value}_{target_regex}"
        f"_n_reps={n_reps}_lgbm_select_seed_compare.parquet"
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


def get_val_score(model: Booster, is_reg: bool) -> float:
    X = model.valid_sets[0].get_data()
    y = model.valid_sets[0].get_label()
    preds = model.predict(X)
    if is_reg:
        return float(explained_variance_score(y, preds))
    if preds.ndim == 1:
        preds = preds.round()
        return balanced_accuracy_score(y, preds)
    preds = np.argmax(preds, axis=1)
    return balanced_accuracy_score(y, preds)


def get_score(model: Booster, X: DataFrame, y: ndarray, is_reg: bool) -> float:
    preds = model.predict(X)
    if is_reg:
        return float(explained_variance_score(y, preds))
    if preds.ndim == 1:
        preds = preds.round()
        return balanced_accuracy_score(y, preds)
    preds = np.argmax(preds, axis=1)
    return balanced_accuracy_score(y, preds)


def _eval_lgb(pargs: LgbmEvalArgs) -> DataFrame:
    callbacks = lightgbm.early_stopping(stopping_rounds=10, verbose=False)

    filterwarnings("ignore", message="Overriding", category=UserWarning)
    filterwarnings("ignore", message=".*n_estimators", category=UserWarning)
    results = lightgbm.cv(
        params=pargs.params,
        train_set=pargs.data_train,
        nfold=5,
        stratified=True,
        callbacks=[callbacks],
        return_cvbooster=True,
    )
    cv_model: CVBooster = results["cvbooster"]
    val_scores = [get_val_score(m, pargs.is_reg) for m in cv_model.boosters]
    cv_score = np.mean(val_scores)
    model = lightgbm.train(
        params=pargs.params,
        train_set=pargs.data_train,
        callbacks=[callbacks],
    )
    test_score = get_score(model, pargs.X_test, pargs.y_test, pargs.is_reg)
    train_score = get_score(model, pargs.X_train, pargs.y_train, pargs.is_reg)
    args = pargs.params.copy()
    for key in pargs.base_params.keys():
        args.pop(key)
    args.pop("n_jobs")

    cols = (
        ["exp_tr", "exp_cv", "exp_test", "args"]
        if pargs.is_reg
        else ["acc_tr", "acc_cv", "acc_test", "args"]
    )
    return DataFrame(
        data=[(train_score, cv_score, test_score, args)],
        columns=cols,
        index=[0],
    )


def lgbm_early_stopping(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    target_regex: str,
    holdout: float | None = 0.25,
    val: float | None = 0.25,
    seed: int | None = None,
) -> DataFrame:
    outfile = OUT / (
        f"{dataset.value}_{feature_regex.value}_{target_regex}"
        f"_seed={seed}_lgbm_select_early_stop_seed_compare.parquet"
    )
    if outfile.exists():
        return pd.read_parquet(outfile)
    is_reg, is_bin, X_train, X_test, y_train, y_test = data_setup(
        dataset=dataset,
        feature_regex=feature_regex,
        target_regex=target_regex,
        holdout=holdout,
        seed=seed,
    )
    data = LGBMDataset(data=X_train, label=y_train, free_raw_data=False)
    data_test = LGBMDataset(data=X_test, label=y_test, free_raw_data=False)
    data_val = ""
    data_train = data
    if val is not None:
        idx_train, idx_val = next(
            StratifiedShuffleSplit(n_splits=1, test_size=val, random_state=seed).split(
                y_train, y_train
            )
        )
        data_train = data.subset(used_indices=idx_train.tolist())
        data_val = data.subset(used_indices=idx_val.tolist())
        data_train.create_valid(data_val)

    objective = "regression" if is_reg else "binary" if is_bin else "multiclass"
    num_class = len(np.unique(y_train)) if objective == "multiclass" else 1
    is_unbalance = not is_reg
    base_params = dict(
        objective=objective,
        num_class=num_class,
        # data=data_train,
        # valid=data_val,
        force_col_wise=True,
        verbosity=-1,
        is_unbalance=is_unbalance,
    )
    all_params = get_params()[:128]
    all_params = [{**base_params, **params} for params in all_params]
    filterwarnings("ignore", message="Overriding", category=UserWarning)
    filterwarnings("ignore", message=".*n_estimators", category=UserWarning)
    all_pargs = [
        LgbmEvalArgs(
            base_params=base_params,
            params=params,
            data_train=data_train,
            is_reg=is_reg,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        for params in all_params
    ]
    dfs: list[DataFrame]
    dfs = Parallel(n_jobs=-1, verbose=10)(delayed(_eval_lgb)(parg) for parg in all_pargs)  # type: ignore  # noqa
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(outfile)
    print(f"Saved results to {outfile}")
    print(df)
    return df


if __name__ == "__main__":
    pd.options.display.max_rows = 1000
    pd.options.display.max_info_rows = 1000
    pd.options.display.max_colwidth = 180

    lgbm_early_stopping(
        dataset=FreesurferStatsDataset.ABIDE_I,
        feature_regex=FeatureRegex.FS,
        # target_regex="int_g_like",
        # target_regex="autism",
        target_regex="MULTI__dsm_iv",
        holdout=0.25,
    )
    # lgbm_feature_select(
    #     dataset=FreesurferStatsDataset.ABIDE_I,
    #     feature_regex=FeatureRegex.FS,
    #     target_regex="MULTI__dsm_iv",
    #     holdout=0.25,
    # )
    sys.exit()

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
