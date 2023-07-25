from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from typing import Mapping, Union, overload

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from src.analysis.feature_select.preprocessing import drop_target_nans, load_preprocessed
from src.analysis.utils import get_kfold
from src.constants import CACHED_RESULTS, PBAR_COLS, PBAR_PAD, TABLES, ensure_dir
from src.enumerables import (
    ClassificationMetric,
    ClassificationModel,
    FeatureRegex,
    FreesurferStatsDataset,
    RegressionMetric,
    RegressionModel,
)
from src.feature_selection.better_stepwise import ForwardSelect
from src.feature_selection.kfold import BinStratifiedKFold

STEPUP_RESULTS = ensure_dir(TABLES / "stepup_results")
FORWARD_RESULTS = ensure_dir(TABLES / "forward_results")
STEPUP_CACHE = ensure_dir(CACHED_RESULTS / "stepup_selection")
FORWARD_CACHE = ensure_dir(CACHED_RESULTS / "forward_selection")


def forward_select_target(
    df_train: DataFrame,
    df_test: DataFrame,
    target: str,
    regex: FeatureRegex,
    model: Union[RegressionModel, ClassificationModel],
    params: Mapping,
    scorer: Union[RegressionMetric, ClassificationMetric],
    max_n_features: int,
    bin_stratify: bool,
    inner_progress: bool,
) -> DataFrame:
    if model in [ClassificationModel.Logistic, ClassificationModel.SVC]:
        params = {**params, **dict()}

    X_train = df_train.filter(regex=regex.value).to_numpy()
    X_test = df_test.filter(regex=regex.value).to_numpy()
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()
    if model is RegressionModel.Lasso:
        X_train = np.asfortranarray(X_train)
        X_test = np.asfortranarray(X_test)

    selector = ForwardSelect(
        estimator=model.get(params),  # type: ignore
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_features_to_select=max_n_features,
        scoring=scorer,
        cv=5 if not bin_stratify else BinStratifiedKFold(5),
        inner_progress=inner_progress,
        n_jobs=-1,
    )
    selector.select()
    fold_info = selector.results

    addons = {
        0: ("source", regex.value),
        1: ("model", model.value),
        2: ("target", str(target).replace("TARGET__", "")),
        3: ("scorer", scorer.value),
    }

    for position, (colname, value) in addons.items():
        fold_info.insert(position, colname, value)
    return fold_info


def nested_stepup_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    models: tuple[RegressionModel, ClassificationModel] = (
        RegressionModel.Lasso,
        ClassificationModel.SGD,
    ),
    scoring: tuple[RegressionMetric, ClassificationMetric] = (
        RegressionMetric.ExplainedVariance,
        ClassificationMetric.BalancedAccuracy,
    ),
    max_n_features: int = 100,
    reg_params: Mapping | None = None,
    cls_params: Mapping | None = None,
    bin_stratify: bool = True,
    inner_progress: bool = False,
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
    means: DataFrame
        DataFrame with columns:

    folds: DataFrame

    """

    if reg_params is None:
        reg_params = dict()
    if cls_params is None:
        cls_params = dict()
    reg_model, cls_model = models
    reg_scoring, cls_scoring = scoring
    regex = feature_regex.value

    r = regex.replace("|", "+")
    fbase = (
        f"{dataset.value}_{reg_model.value}_{cls_model.value}"
        f"_forward_{reg_scoring.value}_{cls_scoring.value}_selected"
        f"_{r}_n={max_n_features}"
    )
    all_fold_out = FORWARD_RESULTS / f"{fbase}_nested_folds.parquet"
    if use_cached and load_complete and all_fold_out.exists():
        return pd.read_parquet(all_fold_out)

    df = load_preprocessed(dataset=dataset, regex=feature_regex, models=models)

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

        model = reg_model if is_reg else cls_model
        params = reg_params if is_reg else cls_params
        scorer = reg_scoring if is_reg else cls_scoring
        df_select = drop_target_nans(df, target)
        cv = get_kfold(bin_stratify=bin_stratify, is_reg=is_reg)
        y = df_select[target]
        all_fold_results: list[DataFrame] = []
        tname = str(target).replace("TARGET__", "")
        fold_out = FORWARD_CACHE / f"{fbase}_{tname}_folds.parquet"

        if fold_out.exists() and use_cached:
            fold_info = pd.read_parquet(fold_out)
        else:
            for outer_fold, (idx_train, idx_test) in enumerate(cv.split(y, y)):
                desc = f"Outer fold {outer_fold + 1}"
                inner_pbar.set_description(f"{desc:>{PBAR_PAD}}")
                df_train, df_test = df_select.iloc[idx_train], df_select.iloc[idx_test]
                fold_results = forward_select_target(
                    df_train=df_train,
                    df_test=df_test,
                    target=target,
                    regex=feature_regex,
                    model=model,  # type: ignore
                    params=params,
                    scorer=scorer,  # type: ignore
                    max_n_features=max_n_features,
                    bin_stratify=bin_stratify,
                    inner_progress=inner_progress,
                )
                fold_results.insert(4, "outer_fold", outer_fold)
                all_fold_results.append(fold_results)
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


def accumulate_features(g: DataFrame) -> DataFrame:
    df = g.copy().reset_index(drop=True)
    df["features"] = ""
    selected = g["selected"].to_list()
    for i in range(len(selected)):
        df.loc[i, "features"] = str(selected[: i + 1])
    return df


def nested_results_to_means(folds: DataFrame) -> DataFrame:
    means = (
        folds.drop(columns=["inner_fold"])
        .groupby(["source", "model", "target", "outer_fold", "selection_iter"])
        .mean(numeric_only=True)
        .reset_index()
    )
    means["selected"] = means["selected"].astype(int)
    df = (
        means.groupby(["source", "model", "target", "outer_fold"])
        .apply(accumulate_features)
        .reset_index(drop=True)
        .drop(columns="selected")
    )
    return df


@overload
def means_to_best_n_feats(
    means: DataFrame,
    metric: RegressionMetric | ClassificationMetric,
    test: bool,
) -> DataFrame:
    ...


@overload
def means_to_best_n_feats(
    means: DataFrame,
    metric: tuple[RegressionMetric, ClassificationMetric],
    test: bool,
) -> tuple[DataFrame, DataFrame]:
    ...


def means_to_best_n_feats(
    means: DataFrame,
    metric: RegressionMetric
    | ClassificationMetric
    | tuple[RegressionMetric, ClassificationMetric],
    test: bool,
) -> DataFrame | tuple[DataFrame, DataFrame]:
    leading_cols = ["source", "model", "target", "outer_fold"]
    t = "test_" if test else ""
    if isinstance(metric, tuple):
        rmetric, cmetric = metric
        rcol = f"{t}{rmetric.value}"
        ccol = f"{t}{cmetric.value}"
        df_r = (
            means.groupby(leading_cols)
            .apply(lambda g: g.nlargest(1, rcol))
            .reset_index(drop=True)
            .sort_values(by=rcol)
        )
        df_c = (
            means.groupby(leading_cols)
            .apply(lambda g: g.nlargest(1, ccol))
            .reset_index(drop=True)
            .sort_values(by=ccol)
        )
        # remove NaN cols
        df_r = df_r[df_r[cmetric.value].isna()].dropna(axis=1)
        df_c = df_c[df_c[rmetric.value].isna()].dropna(axis=1)
        return df_r, df_c

    col = f"{t}{metric.value}"
    return (
        means.groupby(leading_cols)
        .apply(lambda g: g.nlargest(1, col))
        .reset_index(drop=True)
        .sort_values(by=col)
    )


if __name__ == "__main__":
    # os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
    infos = []
    for regex in FeatureRegex:
        infos.append(
            nested_stepup_feature_select(
                dataset=FreesurferStatsDataset.HCP,
                feature_regex=regex,
                max_n_features=50,
                bin_stratify=True,
                inner_progress=True,
                use_cached=True,
            )
        )
    pd.options.display.max_rows = 500
    info = pd.concat(infos, axis=0, ignore_index=True)
    means = nested_results_to_means(info)
    bests = means_to_best_n_feats(
        means, metric=RegressionMetric.ExplainedVariance, test=False
    )
    print(bests.round(3))
    final = (
        bests.groupby(bests.columns.to_list()[:3])
        .mean(numeric_only=True)
        .drop(columns="outer_fold")
        .rename(columns={"selection_iter": "mean_n_iter"})
        .sort_values(by=RegressionMetric.ExplainedVariance.value)
    )
    print(final.round(3))
    sys.exit()
