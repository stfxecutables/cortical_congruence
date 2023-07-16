from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import json
import os
import pickle
import platform
import sys
from argparse import Namespace
from enum import Enum
from hashlib import sha256
from math import ceil
from pathlib import Path
from random import shuffle
from typing import Any, Literal, Mapping, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.constants import CACHED_RESULTS, MEMORY, PLOTS, TABLES, ensure_dir
from src.enumerables import (
    ClassificationMetric,
    ClassificationModel,
    FreesurferStatsDataset,
    RegressionMetric,
    RegressionModel,
)
from src.feature_selection.stepwise import StepwiseSelect
from src.munging.hcp import PhenotypicFocus

if platform.system().lower() == "darwin":
    matplotlib.use("QtAgg")

STEPUP_RESULTS = ensure_dir(TABLES / "stepup_results")
STEPUP_CACHE = ensure_dir(CACHED_RESULTS / "stepup_selection")


class FeatureRegex(Enum):
    FS = "FS"
    CMC = "CMC"
    FS_OR_CMC = "FS|CMC"


@MEMORY.cache(ignore=["df", "inner_progress", "count"], verbose=0)
def select_target(
    df: DataFrame,
    target: str,
    regex: str,
    model: Union[RegressionModel, ClassificationModel],
    params: Mapping,
    scorer: Union[RegressionMetric, ClassificationMetric],
    holdout: float | None,
    max_n_features: int,
    inner_progress: bool,
    count: int,
) -> tuple[DataFrame, float, int]:
    is_reg = "REG" in str(target)
    y = df[target]
    idx_nan = y.isnull()
    df = df[idx_nan].copy()

    if holdout is not None:
        stratify = None if is_reg else y
        df, df_test = train_test_split(df, test_size=holdout, stratify=stratify)
    else:
        df_test = df.copy()
    if model in [ClassificationModel.Logistic, ClassificationModel.SVC]:
        params = {**params, **dict()}

    features = df.filter(regex=regex)
    X = features.to_numpy()
    y = y.to_numpy()
    if model is RegressionModel.Lasso:
        X = np.asfortranarray(X)

    seq = StepwiseSelect(
        n_features_to_select=max_n_features,
        tol=1e-5,
        estimator=model.get(params),  # type: ignore
        direction="forward",
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        inner_progress=inner_progress,
    )
    seq.fit(X, y)
    s = np.array(seq.iteration_scores)
    best = np.max(seq.iteration_scores)
    if scorer in RegressionMetric.inverted():
        best = -best
    n_best = int(np.min(np.where(s == s.max())))
    info = seq.iteration_metrics[n_best]
    info = info.drop(columns=["fit_time", "score_time"])
    info = info.mean()
    if is_reg:
        for reg in RegressionMetric.inverted():
            info[reg.value] = -info[reg.value]
    if holdout is not None:
        idx = seq.iteration_features
        X_test = df_test.filter(regex=regex).iloc[:, idx].to_numpy()
        y_test = df_test[target]
        estimator = model.get(params)
        estimator.fit(features.iloc[:, idx].to_numpy(), y)
        y_pred = estimator.predict(X_test)
        holdout_info = dict()
        for metric in scorer.__class__:
            if (
                isinstance(estimator, RidgeClassifierCV)
                and metric is ClassificationMetric.AUROC
            ):
                continue
            if (
                isinstance(estimator, LogisticRegression)
                and metric is ClassificationMetric.AUROC
            ):
                y_prob = estimator.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    # sklearn auc implementation very bugged
                    y_prob = y_prob[:, 0]
                    auc = float(
                        roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr")
                    )
                    holdout_info[f"test_{metric.value}"] = max(1 - auc, auc)
                else:
                    holdout_info[f"test_{metric.value}"] = metric(y_test, y_prob)
            else:
                holdout_info[f"test_{metric.value}"] = metric(y_test, y_pred)
    else:
        holdout_info = {}
    results = DataFrame(
        {
            "source": regex,
            "model": model.value,
            "target": str(target).replace("TARGET__", ""),
            "scorer": scorer.value,
            "best_score": best,
            **info.to_dict(),
            **holdout_info,
            "n_best": n_best,
            "features": str(seq.iteration_features),
        },
        index=[count],
    )
    return results, best, n_best


def stepup_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    models: tuple[RegressionModel, ClassificationModel] = (
        RegressionModel.Lasso,
        ClassificationModel.Logistic,
    ),
    scoring: tuple[RegressionMetric, ClassificationMetric] = (
        RegressionMetric.MeanAbsoluteError,
        ClassificationMetric.AUROC,
    ),
    max_n_features: int = 100,
    holdout: float | None = 0.25,
    nans: Literal["mean", "drop"] = "mean",
    reg_params: Mapping | None = None,
    cls_params: Mapping | None = None,
    inner_progress: bool = False,
    use_cached: bool = True,
) -> DataFrame:
    """
    Perform stepup feature selection

    Parameters
    ----------
    dataset: FreesurferStatsDataset
        Dataset to use for selection.

    feature_regex: Literal["FS", "CMC", "FS|CMC"]
        Regex to use with `df.filter(regex=feature_regex)`

    models: tuple[RegressionModel, ClassificationModel]
        Models to use for performing feature selection.

    scoring: tuple[RegressionMetric, ClassificationMetric]
        Evaluation metrics for selection.

    max_n_features: int = 100
        Number of features after which to stop selection.

    inner_progress: bool = False
        If True, show a progress bar for the inner loop over the `max_n_features`
        iters.

    holdout: float | None = 0.25
        If a value in (0, 1), evaluate selected features on a random holdout.

    reg_params: Mapping | None = None
        Params to pass to `reg_model`.

    cls_params: Mapping | None = None
        Params to pass to `cls_model`.

    use_cached: bool = True
        If True, load saved results if available.

    Returns
    -------
    scores: DataFrame
        DataFrame with columns:

        source: feature_regex
        model: model for selection
        target: prediction target
        scorer: scoring metric
        best_score: best score on scoring metric in internal k-fold (overfit)
        mae: mean absolute error on internal k-fold
        smae: scaled mean absolute error on internal k-fold
        mad: median absolute deviation on internal k-fold
        smad: scaled median absolute deviation on internal k-fold
        exp-var: explained variance on internal k-fold
        r2: R-squared on internal k-fold
        test_mae: mean absolute error on holdout set
        test_smae: scaled mean absolute error on holdout set
        test_mad: median absolute error on holdout set
        test_smad: scaled median absolute error on holdout set
        test_exp-var: expained variance on holdout set
        test_r2: R-squared on holdout set
        n_best: number of features selected to yield `best_score` column
        features: a string of the form "[f1, f2, ...]" where each fi is the
                  column index of the selected feature at iteration i
    """

    if reg_params is None:
        reg_params = dict()
    if cls_params is None:
        cls_params = dict()
    reg_model, cls_model = models
    reg_scoring, cls_scoring = scoring
    regex = feature_regex.value

    h = f"_holdout={holdout}" if holdout is not None else ""
    r = regex.replace("|", "+")
    fname = (
        f"{dataset.value}_{reg_model.value}_{cls_model.value}"
        f"_stepup_{reg_scoring.value}_{cls_scoring.value}_selected"
        f"_{r}_n={max_n_features}{h}.parquet"
    )
    scores_out = TABLES / fname
    if scores_out.exists() and use_cached:
        return pd.read_parquet(scores_out)

    df = dataset.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )
    if nans == "drop":
        raise NotImplementedError("NaN Dropping not implemented")
    elif nans == "mean":
        feats = df.filter(regex=regex)
        feature_cols = feats.columns
        df.loc[:, feature_cols] = feats.fillna(feats.mean())
    else:
        raise ValueError(f"Undefined nan handling: {nans}")

    if (reg_model is RegressionModel.Lasso) or (
        cls_model in [ClassificationModel.Logistic, ClassificationModel.Ridge]
    ):
        # need to standardize features for co-ordinate descent in LASSO, keep
        # comparisons to Logistic the same
        feature_cols = df.filter(regex=regex).columns
        df.loc[:, feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    all_scores = []
    count = 0

    cols = df.filter(regex="TARGET").columns.to_list()
    pbar = tqdm(cols, leave=True)
    for target in pbar:
        is_reg = "REG" in str(target)
        model = reg_model if is_reg else cls_model
        params = reg_params if is_reg else cls_params
        scorer = reg_scoring if is_reg else cls_scoring
        results, best, n_best = select_target(
            target=target,
            df=df,
            regex=regex,
            model=model,
            params=params,
            scorer=scorer,
            holdout=holdout,
            max_n_features=max_n_features,
            inner_progress=inner_progress,
            count=count,
        )
        all_scores.append(results)
        name = str(target).replace("TARGET__", "")
        metric = (
            f"{np.round(100 * best, 3)}%"
            if scorer is RegressionMetric.ExplainedVariance
            else f"{np.round(best, 4)}"
        )
        pbar.set_description(
            f"{name}: {scorer.value.upper()}={metric} @ {n_best} features"
        )
        count += 1
    scores = pd.concat(all_scores, axis=0)
    scores.to_parquet(scores_out)
    print(f"Saved scores to {scores_out}")
    return scores


def evaluate_HCP_features() -> None:
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select(
            dataset=FreesurferStatsDataset.HCP,
            feature_regex=regex,
            models=(RegressionModel.Lasso, ClassificationModel.SGD),
            scoring=(
                RegressionMetric.MeanAbsoluteError,
                ClassificationMetric.BalancedAccuracy,
            ),
            max_n_features=50,
            holdout=0.25,
            nans="mean",
            reg_params=dict(),
            cls_params=dict(),
            inner_progress=True,
        )
        # scores["source"] = regex
        print(scores.drop(columns=["features"]))
        all_scores.append(scores)
    df = pd.concat(all_scores, axis=0)
    with pd.option_context("display.max_rows", 500):
        print(df.sort_values(by="test_exp-var", ascending=True).round(3))


if __name__ == "__main__":
    # os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    # evaluate_HCP_features()
    # sys.exit()
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select(
            dataset=FreesurferStatsDataset.ABIDE_I,
            feature_regex=regex,
            models=(RegressionModel.Lasso, ClassificationModel.SGD),
            scoring=(
                RegressionMetric.MeanAbsoluteError,
                ClassificationMetric.BalancedAccuracy,
            ),
            max_n_features=50,
            holdout=0.25,
            nans="mean",
            reg_params=dict(),
            cls_params=dict(),
            inner_progress=True,
        )
        # scores["source"] = regex
        print(scores.drop(columns=["features"]))
        all_scores.append(scores)
    df = pd.concat(all_scores, axis=0)
    classification = df[df.scorer == "acc_bal"].dropna(axis=1).drop(columns="features")
    regression = df[df.scorer != "acc_bal"].dropna(axis=1).drop(columns="features")
    with pd.option_context("display.max_rows", 500):
        print(regression.sort_values(by="test_exp-var", ascending=True).round(3))
        print(classification.sort_values(by="test_f1", ascending=True).round(3))

    sys.exit()
    reduce_cmc = False
    reduce_targets = True
    df = FreesurferStatsDataset.HCP.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    sbn.set_style("darkgrid")
    targs = df.filter(regex="TARGET").rename(columns=lambda s: s.replace("TARGET__", ""))
    targs.hist(bins=50, color="black")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.tight_layout()
    fig.savefig(str(PLOTS / "HCP_latent_target_distributions.png"), dpi=300)
    plt.close()

    print(
        targs.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975])
        .round(2)
        .T.to_markdown(tablefmt="simple", floatfmt="0.2f")
    )

    sys.exit()
    # cross_val_score(LGB(max_depth=1, min_data_in_leaf=5, extra_trees=True, max_bin=20, n_jobs=-1), x, y, cv=5, scoring="explained_variance")  # NOTE: for CMC

    scores = [stepup_feature_select(feature_regex=reg) for reg in ["CMC", "FS", "FS|CMC"]]
    scores = pd.concat(scores, axis=0).reset_index(drop=True)
    scores.insert(3, "CMC ratio", 0.0)
    scores["Feat Names"] = ""
    for i in range(scores.shape[0]):
        regex = scores.loc[i, "source"]
        n = int(scores.loc[i, "n_best"])  # type: ignore
        idx = np.array(eval(scores.loc[i, "features"]), dtype=np.int64)
        cols = df.filter(regex=regex).columns.to_numpy()[idx]
        n_cmc = sum("CMC" in col for col in cols)
        if regex == "CMC":
            scores.loc[i, "CMC ratio"] = n_cmc / n
        scores.loc[i, "Feat Names"] = f'"{str(cols)}"'

    scores_out = TABLES / "stepup_selected_scores_linear_all_sources.csv.gz"
    scores.to_csv(scores_out, compression={"method": "gzip", "compresslevel": 9})
    print(f"Saved linear scores to {scores_out}")
    print(scores)
    sys.exit()
    pivoted = (
        scores.drop(columns="features")
        .groupby("source")
        .apply(
            lambda g: g.drop(columns="source").sort_values(
                by=["test_exp_var", "target"], ascending=[False, True]
            )
        )
        .droplevel(1)
        .reset_index()
        .pivot(columns="source", index="target")
    )
    print(
        pivoted.sort_values(
            by=[("Exp.Var", "FS|CMC"), ("Exp.Var", "FS"), ("Exp.Var", "CMC")],
            ascending=False,
        )
        .round(2)
        .rename(
            columns={
                "n_best": "n_selected",
                "Exp.Var": "Explained Variance (%)",
                "CMC ratio": "Proportion CMC Features",
            }
        )
    )

    # features = df.filter(regex=feature_regex)

    # pd.options.display.max_rows = 500
    # load_HCP_complete.clear()
