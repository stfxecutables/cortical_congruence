from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import ParameterGrid, cross_validate
from tqdm import tqdm

from src.analysis.feature_select.preprocessing import drop_target_nans, load_preprocessed
from src.analysis.utils import get_kfold
from src.constants import CACHED_RESULTS, PBAR_COLS, PBAR_PAD, TABLES, ensure_dir
from src.enumerables import (
    ClassificationMetric,
    FeatureRegex,
    FreesurferStatsDataset,
    RegressionMetric,
    RegressionModel,
)

LGBM_RESULTS = ensure_dir(TABLES / "lgbm_results")
LGBM_CACHE = ensure_dir(CACHED_RESULTS / "lgbm_selection")

DEFAULTS = dict(
    # boosting_type="rf",
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    # subsample=1.0,
    n_estimators=100,
    reg_alpha=0.0,
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class FitArgs:
    lgb_args: Mapping
    X_train: ndarray
    X_test: ndarray
    y_train: ndarray
    y_test: ndarray
    is_reg: bool
    bin_stratify: bool
    iter: int


def fit_lgb(fit_args: FitArgs) -> DataFrame:
    defaults = DEFAULTS

    param = fit_args.lgb_args
    is_reg = fit_args.is_reg
    X_train = fit_args.X_train
    X_test = fit_args.X_test
    y_train = fit_args.y_train
    y_test = fit_args.y_test
    bin_stratify = fit_args.bin_stratify
    tune_iter = fit_args.iter

    args = {**defaults, **param}

    estimator = LGBMRegressor(**args) if is_reg else LGBMClassifier(**args)
    metric_cls = RegressionMetric if is_reg else ClassificationMetric
    scorers = metric_cls.scorers()
    cv = get_kfold(bin_stratify=bin_stratify, is_reg=is_reg)
    results = DataFrame(
        cross_validate(
            estimator=estimator,  # type: ignore
            X=X_train,
            y=y_train,
            scoring=scorers,
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            return_estimator=True,
        )
    )
    estimators = results["estimator"].to_list()
    imps = [json.dumps(est.feature_importances_.tolist()) for est in estimators]
    results.drop(columns=["estimator", "fit_time", "score_time"], inplace=True)
    if is_reg:
        for metric in RegressionMetric.inverted():
            train = f"train_{metric.value}"
            test = f"test_{metric.value}"
            results[train] *= -1
            results[test] *= -1
    results.rename(columns=lambda s: f"inner_{s}", inplace=True)

    estimator = LGBMRegressor(**args) if is_reg else LGBMClassifier(**args)
    estimator.fit(X_train, y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)

    for metric in metric_cls:
        results[f"outer_train_{metric.value}"] = metric(y_train, y_pred_train)
    for metric in metric_cls:
        results[f"outer_test_{metric.value}"] = metric(y_test, y_pred_test)

    results["params"] = json.dumps(param)
    results["inner_importances"] = imps
    results["outer_importances"] = json.dumps(estimator.feature_importances_.tolist())
    results.insert(0, "inner_fold", range(1, 6))
    results.insert(1, "tune_iter", tune_iter)

    return results


def get_params(log: bool = False) -> list[dict]:
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
                # subsample=[0.05, 0.2, 0.4, 0.6, 0.8, 1.0],
                subsample=[0.05, 0.2, 0.4, 0.6, 0.8, 0.9],  # for rf
                subsample_freq=[1],  # for rf
                n_estimators=[100],
                reg_alpha=np.logspace(-3, 2, num=10, base=10).tolist(),
                n_jobs=[1],
            )
        )
    )
    params = params[np.random.permutation(len(params))]
    if log:
        print(f"n_params: {len(params)}")
    return params.tolist()


def lgbm_select_target(
    df_train: DataFrame,
    df_test: DataFrame,
    target: str,
    regex: FeatureRegex,
    scoring: tuple[RegressionMetric, ClassificationMetric],
    bin_stratify: bool,
    n_tune: int = 8,
) -> DataFrame:
    feats = np.array(df_train.filter(regex=regex.value).columns.to_list())
    X_train = df_train.filter(regex=regex.value).to_numpy()
    X_test = df_test.filter(regex=regex.value).to_numpy()
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()

    X_train = np.asfortranarray(X_train)
    X_test = np.asfortranarray(X_test)

    is_reg = "REG" in target

    params = get_params()
    params = params[:n_tune]
    args = [
        FitArgs(
            lgb_args=param,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            is_reg=is_reg,
            bin_stratify=bin_stratify,
            iter=tune_iter,
        )
        for tune_iter, param in enumerate(params)
    ]

    fits: list[DataFrame]
    # results = [fit_lgb(arg) for arg in args]  # debug
    fits = Parallel(n_jobs=-1, verbose=0)(delayed(fit_lgb)(arg) for arg in args)  # type: ignore  # noqa
    tune_results = pd.concat(fits, axis=0, ignore_index=True)
    means = (
        tune_results.groupby("tune_iter")
        .mean(numeric_only=True)
        .drop(columns="inner_fold")
    )
    reg_sort = f"inner_test_{scoring[0].value}"
    cls_sort = f"inner_test_{scoring[1].value}"
    sorter = reg_sort if is_reg else cls_sort

    best_idx = means.sort_values(by=sorter, ascending=is_reg).head(10).index
    results = tune_results[tune_results["tune_iter"].isin(best_idx)]

    addons = {
        0: ("source", regex.value),
        1: ("target", str(target).replace("TARGET__", "")),
    }

    for position, (colname, value) in addons.items():
        results.insert(position, colname, value)

    return results


def nested_lgbm_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    scoring: tuple[RegressionMetric, ClassificationMetric] = (
        RegressionMetric.ExplainedVariance,
        ClassificationMetric.BalancedAccuracy,
    ),
    n_tune: int = 128,
    bin_stratify: bool = True,
    use_cached: bool = True,
    load_complete: bool = False,
) -> DataFrame:
    """
    Perform stepup feature selection

    Parameters
    ----------
    dataset: FreesurferStatsDataset
        Dataset to use for selection.

    feature_regex: Literal["FS", "CMC", "FS|CMC"]
        Regex to use with `df.filter(regex=feature_regex)`

    scoring: RegressionMetric
        Evaluation metrics for selection.

    inner_progress: bool = False
        If True, show a progress bar for the inner loop over the `max_n_features`
        iters.

    holdout: float | None = 0.25
        If a value in (0, 1), evaluate selected features on a random holdout.

    use_cached: bool = True
        If True, load saved results if available.

    Returns
    -------
    means: DataFrame
        DataFrame with columns:

    folds: DataFrame

    """

    regex = feature_regex.value

    r = regex.replace("|", "+")
    fbase = f"{dataset.value}_{r}_lgbm_selected_htune={n_tune}"
    all_fold_out = LGBM_RESULTS / f"{fbase}_nested.parquet"
    if use_cached and load_complete and all_fold_out.exists():
        return pd.read_parquet(all_fold_out)

    df = load_preprocessed(
        dataset=dataset, regex=feature_regex, models=RegressionModel.LightGBM
    )

    all_fold_infos = []

    pbar = tqdm(
        df.filter(regex="TARGET").columns.to_list(),
        leave=True,
        desc=f"{'Target loop':>{PBAR_PAD}}",
        ncols=PBAR_COLS,
    )
    inner_pbar = tqdm(
        desc=f"{'Outer fold':>{PBAR_PAD}}", total=5, leave=True, ncols=PBAR_COLS
    )
    for target in pbar:
        is_reg = "REG" in str(target)

        df_select = drop_target_nans(df, target)
        cv = get_kfold(bin_stratify=bin_stratify, is_reg=is_reg)
        y = df_select[target]
        all_fold_results: list[DataFrame] = []
        tname = str(target).replace("TARGET__", "")
        fold_out = LGBM_CACHE / f"{fbase}_{tname}_folds.parquet"

        if fold_out.exists() and use_cached:
            fold_info = pd.read_parquet(fold_out)
        else:
            for outer_fold, (idx_train, idx_test) in enumerate(cv.split(y, y)):
                desc = f"Outer fold {outer_fold + 1}"
                inner_pbar.set_description(f"{desc:>{PBAR_PAD}}")
                df_train, df_test = df_select.iloc[idx_train], df_select.iloc[idx_test]
                results = lgbm_select_target(
                    df_train=df_train,
                    df_test=df_test,
                    target=target,
                    regex=feature_regex,
                    scoring=scoring,
                    n_tune=n_tune,
                    bin_stratify=bin_stratify,
                )
                results.insert(2, "outer_fold", outer_fold + 1)
                all_fold_results.append(results.copy())
                inner_pbar.update()
            inner_pbar.reset()

            fold_info = pd.concat(all_fold_results, axis=0, ignore_index=True)
            fold_info.to_parquet(fold_out)

        all_fold_infos.append(fold_info)
    pbar.close()

    all_fold_info = pd.concat(all_fold_infos, axis=0, ignore_index=True)
    all_fold_info.to_parquet(all_fold_out)
    print(f"Saved nested forward selection results to: {all_fold_out}")

    return all_fold_info


if __name__ == "__main__":
    data = FreesurferStatsDataset.HCP
    infos = []
    for regex in FeatureRegex:
        infos.append(
            nested_lgbm_feature_select(
                dataset=data,
                feature_regex=regex,
                bin_stratify=True,
                use_cached=True,
                n_tune=128,
            )
        )
    info = pd.concat(infos, axis=0, ignore_index=True)
    bests = (
        info.groupby(["source", "target", "outer_fold"])
        .mean(numeric_only=True)
        .filter(regex="exp-var|smae")
        .reset_index()
        .groupby(["source", "target"])
        .mean()
        .sort_values(by="inner_test_exp-var")
        .drop(columns="outer_fold")
        .reset_index()
    )

    print(bests.round(4).to_markdown(tablefmt="simple", index=False, floatfmt="0.4f"))

    # see a lot of https://stats.stackexchange.com/questions/319861/how-to-interpret-lasso-shrinking-all-coefficients-to-0
    # basically, we don't have linear relationships
    dfs = []
    for _, df in info.groupby(["source", "target"]):
        if all(df["inner_selected"] == "none") and all(df["outer_selected"] == "none"):
            continue
        dfs.append(df)
    sel = pd.concat(dfs, axis=0)
    bests = (
        sel.groupby(["source", "target", "outer_fold"])
        .mean(numeric_only=True)
        .filter(regex="exp-var")
        .reset_index()
        .groupby(["source", "target"])
        .mean()
        .sort_values(by="inner_test_exp-var")
        .drop(columns="outer_fold")
        .reset_index()
    )
    print(bests.round(4).to_markdown(tablefmt="simple", index=False, floatfmt="0.4f"))
