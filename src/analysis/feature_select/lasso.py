from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_validate
from tqdm import tqdm

from src.analysis.feature_select.preprocessing import drop_target_nans, load_preprocessed
from src.analysis.utils import get_kfold
from src.constants import CACHED_RESULTS, PBAR_COLS, PBAR_PAD, TABLES, ensure_dir
from src.enumerables import (
    FeatureRegex,
    FreesurferStatsDataset,
    RegressionMetric,
    RegressionModel,
)

STEPUP_RESULTS = ensure_dir(TABLES / "stepup_results")
FORWARD_RESULTS = ensure_dir(TABLES / "forward_results")
STEPUP_CACHE = ensure_dir(CACHED_RESULTS / "stepup_selection")
FORWARD_CACHE = ensure_dir(CACHED_RESULTS / "forward_selection")


def lasso_select_target(
    df_train: DataFrame,
    df_test: DataFrame,
    target: str,
    regex: FeatureRegex,
    bin_stratify: bool,
) -> DataFrame:
    feats = np.array(df_train.filter(regex=regex.value).columns.to_list())
    X_train = df_train.filter(regex=regex.value).to_numpy()
    X_test = df_test.filter(regex=regex.value).to_numpy()
    y_train = df_train[target].to_numpy()
    y_test = df_test[target].to_numpy()

    X_train = np.asfortranarray(X_train)
    X_test = np.asfortranarray(X_test)

    cv = get_kfold(bin_stratify=bin_stratify, is_reg=True)
    estimator = RegressionModel.Lasso.get()
    results = DataFrame(
        cross_validate(
            estimator,
            X_train,
            y_train,
            scoring=RegressionMetric.scorers(),
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            return_estimator=True,
        )
    )
    estimators = results["estimator"].to_list()
    results.drop(columns=["estimator", "fit_time", "score_time"], inplace=True)
    for metric in RegressionMetric.inverted():
        train = f"train_{metric.value}"
        test = f"test_{metric.value}"
        results[train] *= -1
        results[test] *= -1
    results.rename(columns=lambda s: f"inner_{s}", inplace=True)

    tuned_alphas = [est.alpha_ for est in estimators]
    # each of w has shape (X_train.shape[1],)
    ws = [est.coef_ for est in estimators]
    eps = 1e-15  # we should probably use np.spacing, or sthg...
    selected_feats = []
    for w in ws:
        if np.sum(w) < eps:
            selected_feats.append("none")
        else:
            idx = np.abs(w) > eps
            selected_feats.append(str(feats[idx]))

    estimator = RegressionModel.Lasso.get()  # type: ignore
    estimator.fit(X_train, y_train)
    y_pred_test = estimator.predict(X_test)
    y_pred_train = estimator.predict(X_train)
    for metric in RegressionMetric:
        results[f"outer_train_{metric.value}"] = metric(y_train, y_pred_train)
    for metric in RegressionMetric:
        results[f"outer_test_{metric.value}"] = metric(y_test, y_pred_test)

    outer_alpha = estimator.alpha_
    outer_w = estimator.coef_
    if np.sum(outer_w) < eps:
        outer_selected_feats = "none"
    else:
        outer_idx = np.abs(outer_w) > eps
        outer_selected_feats = str(feats[outer_idx])

    addons = {
        0: ("source", regex.value),
        1: ("target", str(target).replace("TARGET__", "")),
    }

    for position, (colname, value) in addons.items():
        results.insert(position, colname, value)

    results["inner_alpha"] = tuned_alphas
    # results["inner_w"] = ws
    results["inner_selected"] = selected_feats
    results["outer_alpha"] = outer_alpha
    # results["outer_w"] = Series([outer_w])
    results["outer_selected"] = outer_selected_feats
    return results


def nested_lasso_feature_select(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
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
    fbase = f"{dataset.value}_{r}_lasso_selected"
    all_fold_out = FORWARD_RESULTS / f"{fbase}_nested.parquet"
    if use_cached and load_complete and all_fold_out.exists():
        return pd.read_parquet(all_fold_out)

    df = load_preprocessed(
        dataset=dataset, regex=feature_regex, models=RegressionModel.Lasso
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
        fold_out = FORWARD_CACHE / f"{fbase}_{tname}_folds.parquet"

        if fold_out.exists() and use_cached:
            fold_info = pd.read_parquet(fold_out)
        else:
            for outer_fold, (idx_train, idx_test) in enumerate(cv.split(y, y)):
                desc = f"Outer fold {outer_fold + 1}"
                inner_pbar.set_description(f"{desc:>{PBAR_PAD}}")
                df_train, df_test = df_select.iloc[idx_train], df_select.iloc[idx_test]
                results = lasso_select_target(
                    df_train=df_train,
                    df_test=df_test,
                    target=target,
                    regex=feature_regex,
                    bin_stratify=bin_stratify,
                )
                results.insert(2, "outer_fold", outer_fold + 1)
                results.insert(3, "inner_fold", range(1, 6))
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
            nested_lasso_feature_select(
                dataset=data,
                feature_regex=regex,
                bin_stratify=True,
                use_cached=True,
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
