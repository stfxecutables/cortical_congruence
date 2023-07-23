from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import platform
import sys
from pathlib import Path
from typing import Literal, Mapping, Union

import matplotlib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.constants import CACHED_RESULTS, MEMORY, PBAR_COLS, PBAR_PAD, TABLES, ensure_dir
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
from src.munging.hcp import PhenotypicFocus

STEPUP_RESULTS = ensure_dir(TABLES / "stepup_results")
FORWARD_RESULTS = ensure_dir(TABLES / "forward_results")
STEPUP_CACHE = ensure_dir(CACHED_RESULTS / "stepup_selection")
FORWARD_CACHE = ensure_dir(CACHED_RESULTS / "forward_selection")


def load_preprocessed(
    dataset: FreesurferStatsDataset,
    regex: FeatureRegex,
    models: tuple[RegressionModel, ClassificationModel],
) -> DataFrame:
    df = dataset.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )
    feats = df.filter(regex=regex.value)
    feature_cols = feats.columns
    df.loc[:, feature_cols] = feats.fillna(feats.mean())

    reg_model, cls_model = models
    if (reg_model is RegressionModel.Lasso) or (
        cls_model in [ClassificationModel.Logistic, ClassificationModel.Ridge]
    ):
        # need to standardize features for co-ordinate descent in LASSO, keep
        # comparisons to Logistic the same
        df.loc[:, feature_cols] = StandardScaler().fit_transform(df[feature_cols])
    return df


def drop_target_nans(df: DataFrame, target: str) -> DataFrame:
    df = df.copy()
    y = df[target]
    idx_nan = y.isnull()
    df = df[~idx_nan]
    return df


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
    df = df[~idx_nan].copy()
    y = y[~idx_nan]

    if holdout is not None:
        stratify = None if is_reg else y
        df, df_test = train_test_split(df, test_size=holdout, stratify=stratify)
    else:
        df_test = df.copy()
    if model in [ClassificationModel.Logistic, ClassificationModel.SVC]:
        params = {**params, **dict()}

    features = df.filter(regex=regex)
    X = features.to_numpy()
    y = df[target].to_numpy()
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


def stepup_feature_select_holdout(
    dataset: FreesurferStatsDataset,
    feature_regex: FeatureRegex,
    models: tuple[RegressionModel, ClassificationModel] = (
        RegressionModel.Lasso,
        ClassificationModel.Logistic,
    ),
    scoring: tuple[RegressionMetric, ClassificationMetric] = (
        RegressionMetric.MeanAbsoluteError,
        ClassificationMetric.BalancedAccuracy,
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
        if is_reg and bin_stratify:
            cv = BinStratifiedKFold()
        elif is_reg:
            cv = KFold()
        else:
            cv = StratifiedKFold()
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
                    model=model,
                    params=params,
                    scorer=scorer,
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
    all_fold_out = FORWARD_RESULTS / f"{fbase}_nested_folds.parquet"
    all_fold_info.to_parquet(all_fold_out)
    print(f"Saved nested forward selection results to: {all_fold_out}")

    return all_fold_info


def nested_results_to_means(folds: DataFrame) -> DataFrame:
    def accumulate_features(g: DataFrame) -> DataFrame:
        df = g.copy().reset_index(drop=True)
        df["features"] = ""
        selected = g["selected"].to_list()
        for i in range(len(selected)):
            df.loc[i, "features"] = str(selected[: i + 1])
        return df

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


def means_to_best_n_feats(
    means: DataFrame, metric: RegressionMetric | ClassificationMetric
) -> DataFrame:
    leading_cols = ["source", "model", "target", "outer_fold"]
    return (
        means.groupby(leading_cols)
        .apply(lambda g: g.nlargest(1, metric.value))
        .reset_index(drop=True)
    )


def evaluate_HCP_features(n_features: int) -> DataFrame:
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select_holdout(
            dataset=FreesurferStatsDataset.HCP,
            feature_regex=regex,
            models=(RegressionModel.Lasso, ClassificationModel.SGD),
            scoring=(
                RegressionMetric.MeanAbsoluteError,
                ClassificationMetric.BalancedAccuracy,
            ),
            max_n_features=n_features,
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
    return df


def evaluate_ABIDE_I_features() -> None:
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select_holdout(
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
            use_cached=False,
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


def evaluate_ABIDE2_features() -> None:
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select_holdout(
            dataset=FreesurferStatsDataset.ABIDE_II,
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
            use_cached=False,
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


def evaluate_ADHD200_features() -> None:
    all_scores = []
    for regex in FeatureRegex:
        scores = stepup_feature_select_holdout(
            dataset=FreesurferStatsDataset.ADHD_200,
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
            use_cached=False,
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
    bests = means_to_best_n_feats(means, metric=RegressionMetric.ExplainedVariance)
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

    df20 = evaluate_HCP_features(n_features=20)
    df21 = evaluate_HCP_features(n_features=21)
    df50 = evaluate_HCP_features(n_features=50)
    df51 = evaluate_HCP_features(n_features=51)
    for df, n_feat in zip([df20, df21, df50, df51], [20, 21, 50, 51]):
        df["features"] = df["features"].apply(lambda f: sorted(eval(f)))
        print(f"Selecting up to {n_feat} features:")
        corrs = df.loc[
            :, ["mae", "test_mae", "exp-var", "test_exp-var", "smae", "test_smae"]
        ].corr()
        print(f"MAE/MAE_test correlation:   {corrs.loc['mae', 'test_mae'].round(3): .3f}")
        print(
            f"sMAE/sMAE_test correlation: {corrs.loc['smae', 'test_smae'].round(3): .3f}"
        )
        print(
            f"Exp/Exp_test correlation:   {corrs.loc['exp-var', 'test_exp-var'].round(3): .3f}"
        )
        print(
            df.sort_values(by="exp-var", ascending=False)
            .head(10)
            .round(3)
            .loc[
                :,
                [
                    "source",
                    "target",
                    "smae",
                    "test_smae",
                    "mae",
                    "test_mae",
                    "exp-var",
                    "test_exp-var",
                    "features",
                ],
            ]
        )

    # evaluate_ABIDE_I_features()
    # evaluate_ABIDE2_features()
    # evaluate_ADHD200_features()
    sys.exit()
