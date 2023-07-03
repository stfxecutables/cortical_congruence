from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import re
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from io import StringIO
from math import ceil
from numbers import Integral
from pathlib import Path
from random import shuffle
from typing import Any, Callable, Iterable, Literal, cast

import numpy as np
import pandas as pd
import tabulate
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor as LGB
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.errors import ParserError
from sklearn.base import BaseEstimator, _fit_context, clone, is_classifier
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import (
    BaseCrossValidator,
    ParameterGrid,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.model_selection._split import check_cv
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import (
    ABIDE_II_ENCODING,
    ALL_STATSFILES,
    CACHED_RESULTS,
    DATA,
    HCP_FEATURE_INFO,
    MEMORY,
    TABLES,
)
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus
from src.munging.fs_stats import load_HCP_complete

from sklearn.model_selection._split import BaseShuffleSplit  # isort: skip


@dataclass
class SelectArgs:
    estimator: Any
    X: ndarray
    y: ndarray
    mask: ndarray
    idx: ndarray
    direction: str
    scoring: Any
    cv: Any


def _get_score(args: SelectArgs) -> float:
    current_mask = args.mask
    feature_idx = args.idx
    estimator = args.estimator
    direction = args.direction
    X = args.X
    y = args.y
    cv = args.cv
    scoring = args.scoring

    candidate_mask = current_mask.copy()
    candidate_mask[feature_idx] = True
    if direction == "backward":
        candidate_mask = ~candidate_mask
    X_new = X[:, candidate_mask]
    return cross_val_score(
        estimator,
        X_new,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    ).mean()


class StepUpSelect(SequentialFeatureSelector):
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_features_to_select: float | int | Literal["auto", "warn"] = "warn",
        tol: float | None = None,
        direction: Literal["forward", "backward"] = "forward",
        scoring: str | Callable[..., Any] | None = None,
        cv: Iterable | int | BaseShuffleSplit | BaseCrossValidator = 5,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        self.iteration_scores: list[float] = []
        self.iteration_features: list[Any] = []

    @_fit_context(
        # SequentialFeatureSelector.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Learn the features to select from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]

        if self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features - 1
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if self.n_features_to_select >= n_features:
                raise ValueError("n_features_to_select must be < n_features.")
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"
        for _ in range(n_iterations):
            new_feature_idx, new_score = self._get_best_new_feature_score(
                cloned_estimator, X, y, cv, current_mask
            )
            self.iteration_scores.append(new_score)
            self.iteration_features.append(new_feature_idx)
            if is_auto_select and ((new_score - old_score) < self.tol):
                break

            old_score = new_score
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self

    def _get_best_new_feature_score(self, estimator, X, y, cv, current_mask):
        # Return the best new feature and its score to add to the current_mask,
        # i.e. return the best new feature and its score to add (resp. remove)
        # when doing forward selection (resp. backward selection).
        # Feature will be added if the current score and past score are greater
        # than tol when n_feature is auto,
        candidate_feature_indices = np.flatnonzero(~current_mask)
        args = [
            SelectArgs(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                mask=current_mask.copy(),
                idx=feature_idx,
                scoring=self.scoring,
                direction=self.direction,
            )
            for feature_idx in candidate_feature_indices
        ]

        results = Parallel(n_jobs=-1, verbose=0)(delayed(_get_score)(arg) for arg in args)

        scores = {}
        for feature_idx, score in zip(candidate_feature_indices, results):
            scores[feature_idx] = score

        # for feature_idx in tqdm(candidate_feature_indices, leave=True):
        #     candidate_mask = current_mask.copy()
        #     candidate_mask[feature_idx] = True
        #     if self.direction == "backward":
        #         candidate_mask = ~candidate_mask
        #     X_new = X[:, candidate_mask]
        #     scores[feature_idx] = cross_val_score(
        #         estimator,
        #         X_new,
        #         y,
        #         cv=cv,
        #         scoring=self.scoring,
        #         n_jobs=self.n_jobs,
        #     ).mean()
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        return new_feature_idx, scores[new_feature_idx]


def print_correlations() -> None:
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=True
    )
    dfr = df.filter(regex="CMC|TARGET|DEMO")
    corrs = (
        dfr.corr()
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    )
    corrs["abs"] = corrs["r"].abs()
    corrs = corrs.sort_values(by="abs", ascending=False).drop(columns="abs")
    corrs = corrs[corrs["r"] < 1.0]
    cmc_corrs = corrs[
        (corrs.x.str.contains("CMC") & corrs.y.str.contains("TARGET"))
        | (corrs.x.str.contains("TARGET") & corrs.y.str.contains("CMC"))
    ]
    with pd.option_context("display.max_rows", 500):
        print(cmc_corrs)


def compute_regression_metrics(args: Namespace) -> DataFrame:
    dfr = args.data
    model = args.model
    modelname = args.modelname
    target = args.target
    rand = args.random_feats
    perc_max = args.perc_max
    x = dfr.filter(regex=args.feature_regex)
    if rand:
        idx = np.random.permutation(x.shape[1])
        n = np.random.randint(5, max(6, ceil(perc_max * x.shape[1])))
        x = x.iloc[:, idx].iloc[:, :n]
    else:
        n = x.shape[1]

    x = x.to_numpy()
    y = dfr[target]
    results = cross_validate(
        model,
        x,
        y,
        cv=5,
        scoring=["neg_mean_absolute_error", "r2", "explained_variance"],
    )
    df = DataFrame(results).mean().to_frame()
    df = df.T.copy()
    df.rename(
        columns={
            "test_neg_mean_absolute_error": "MAE",
            "test_r2": "R2",
            "test_explained_variance": "Exp.Var",
        },
        inplace=True,
    )
    df.drop(columns=["fit_time", "score_time"], inplace=True)
    df.loc[:, "MAE"] = df["MAE"].abs()
    df.loc[:, "R2"] = df["R2"].abs()
    df.loc[:, "Exp.Var"] = df["Exp.Var"].abs()
    df["target"] = str(target).replace("TARGET__", "")
    df["model"] = modelname
    df["n_feats"] = n
    return df


def get_random_subset_args(
    feature_regex: str = "CMC",
    reduce_cmc: bool = False,
    reduce_targets: bool = True,
    random_feats: bool = False,
    n_random: int = 250,
    perc_max: float = 0.25,
) -> list[Namespace]:
    n_jobs = 1
    models = {
        "LR": lambda: LR(n_jobs=n_jobs),
        # "MLP": lambda: MLP(
        #     hidden_layer_sizes=[32, 64, 128, 256],
        #     activation="relu",
        #     solver="adam",
        #     alpha=1e-4,
        #     shuffle=True,
        #     learning_rate_init=3e-4,
        #     max_iter=500,
        # ),
        # "LGB": lambda: LGB(n_jobs=n_jobs),
        # "KNN-1": lambda: KNN(n_neighbors=1, n_jobs=n_jobs),
        # "KNN-3": lambda: KNN(n_neighbors=3, n_jobs=n_jobs),
        # "KNN-9": lambda: KNN(n_neighbors=9, n_jobs=n_jobs),
        "Dummy": lambda: Dummy(strategy="mean"),
    }
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    dfr = df.copy()
    args = []
    for target in dfr.filter(regex="TARGET").columns:
        for model in models:
            if not random_feats:
                args.append(
                    Namespace(
                        **dict(
                            data=dfr,
                            model=models[model](),
                            modelname=model,
                            target=target,
                            feature_regex=feature_regex,
                            random_feats=False,
                            perc_max=perc_max,
                        )
                    )
                )
            else:
                for _ in range(n_random):
                    args.append(
                        Namespace(
                            **dict(
                                data=dfr,
                                model=models[model](),
                                modelname=model,
                                target=target,
                                feature_regex=feature_regex,
                                random_feats=True,
                                perc_max=perc_max,
                            )
                        )
                    )

    shuffle(args)
    return args


def get_stepup_args(
    feature_regex: str = "CMC",
    reduce_cmc: bool = False,
    reduce_targets: bool = True,
    random_feats: bool = False,
    n_random: int = 250,
    perc_max: float = 0.25,
) -> list[Namespace]:
    n_jobs = 1
    models = {
        "LR": lambda: LR(n_jobs=n_jobs),
        # "MLP": lambda: MLP(
        #     hidden_layer_sizes=[32, 64, 128, 256],
        #     activation="relu",
        #     solver="adam",
        #     alpha=1e-4,
        #     shuffle=True,
        #     learning_rate_init=3e-4,
        #     max_iter=500,
        # ),
        # "LGB": lambda: LGB(n_jobs=n_jobs),
        # "KNN-1": lambda: KNN(n_neighbors=1, n_jobs=n_jobs),
        # "KNN-3": lambda: KNN(n_neighbors=3, n_jobs=n_jobs),
        # "KNN-9": lambda: KNN(n_neighbors=9, n_jobs=n_jobs),
        "Dummy": lambda: Dummy(strategy="mean"),
    }
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    dfr = df.copy()
    args = []
    for target in dfr.filter(regex="TARGET").columns:
        for model in models:
            if not random_feats:
                args.append(
                    Namespace(
                        **dict(
                            data=dfr,
                            model=models[model](),
                            modelname=model,
                            target=target,
                            feature_regex=feature_regex,
                            random_feats=False,
                            perc_max=perc_max,
                        )
                    )
                )
            else:
                for _ in range(n_random):
                    args.append(
                        Namespace(
                            **dict(
                                data=dfr,
                                model=models[model](),
                                modelname=model,
                                target=target,
                                feature_regex=feature_regex,
                                random_feats=True,
                                perc_max=perc_max,
                            )
                        )
                    )

    shuffle(args)
    return args


def sample_random_subsets(use_mean: bool) -> None:
    summaries = []
    for regex in ["FS", "CMC", "FS|CMC"]:
        reduce_cmc = False
        feature_regex = regex
        random_feats = True
        n_random = 5000
        perc_max = 0.3

        args = get_random_subset_args(
            feature_regex=feature_regex,
            reduce_cmc=reduce_cmc,
            random_feats=random_feats,
            n_random=n_random,
            perc_max=perc_max,
        )
        r = "reduced" if reduce_cmc else "raw"
        f = feature_regex.lower()
        if "|" in r:
            r = "+".join(sorted(r.split("|")))
        rnd = f"_rand_{n_random}_of{perc_max:0.2f}p" if random_feats else ""
        out = TABLES / f"quick_regression_cluster_targets_{f}_{r}{rnd}.parquet"
        if out.exists():
            df = pd.read_parquet(out)
        else:
            print(f"Computing {len(args)} iterations:")
            all_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(compute_regression_metrics)(arg) for arg in args
            )
            df = pd.concat(all_results, axis=0).reset_index(drop=True)
            print(df.to_markdown(tablefmt="simple", floatfmt="0.4f"))
            df.to_parquet(out)
            print(f"Saved results to {out}")

        dummy_maes = (
            df.groupby("target")
            .apply(lambda grp: grp[grp.model == "Dummy"]["MAE"])
            .droplevel(1)
            .drop_duplicates()
        )
        base_mae = df.target.apply(lambda target: dummy_maes[target])
        df.insert(3, "sMAE", df["MAE"] / base_mae)
        best = df[df["sMAE"] < 1.0]
        print(
            best.sort_values(by="sMAE", ascending=True)
            .iloc[:200]
            .sort_values(by="sMAE", ascending=False)
            .to_markdown(tablefmt="simple", index=False, floatfmt="0.4f")
        )
        target_bests = (
            df[["target", "Exp.Var", "sMAE", "n_feats"]]
            .groupby("target")
            .apply(lambda g: g.nsmallest(10 if use_mean else 1, columns=["sMAE"]))
            .droplevel(1)
            .drop(columns="target")
            .sort_values(by=["target", "sMAE"])
        )
        if use_mean:
            target_bests = (
                target_bests.groupby("target").mean().sort_values(by=["target", "sMAE"])
            )

        print("=" * 90)
        print(out.name)
        print("=" * 90)
        print(
            target_bests.to_markdown(
                tablefmt="simple", floatfmt=["", "0.5f", "0.4f", "0.0f"]
            )
        )

        target_bests["Exp.Var"] = target_bests["Exp.Var"] * 100
        target_bests = target_bests.rename(columns={"Exp.Var": "Exp.Var (%)"})
        label = regex.replace("|", "+")
        summaries.append(target_bests.rename(columns=lambda s: f"{label}_{s}"))

    summary = pd.concat(summaries, axis=1)
    order = [
        "FS_Exp.Var (%)",
        "CMC_Exp.Var (%)",
        "FS+CMC_Exp.Var (%)",
        "FS_sMAE",
        "CMC_sMAE",
        "FS+CMC_sMAE",
        "FS_n_feats",
        "CMC_n_feats",
        "FS+CMC_n_feats",
    ]
    summary = summary[order].copy()
    for col in summary.filter(regex="n_feats").columns:
        summary[col] = summary[col].astype(str)

    summary = summary.sort_values(
        by=["FS_sMAE", "CMC_sMAE", "FS+CMC_sMAE"], ascending=True
    )
    print(
        summary.to_markdown(
            tablefmt="simple",
            floatfmt=[
                "",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.1f",
                "0.1f",
                "0.1f",
            ],
        )
    )
    summary_out = TABLES / "random_subset_summary.csv.gz"
    summary.to_csv(summary_out, compression={"method": "gzip", "compresslevel": 9})
    print(f"Saved feature subset summary to {summary_out}")


def stepup_feature_select() -> DataFrame:
    scores_out = TABLES / "stepup_selected_scores.parquet"
    if scores_out.exists():
        return pd.read_parquet(scores_out)

    reduce_cmc = False
    reduce_targets = True
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    all_scores = []
    count = 0
    pbar_outer = tqdm(["CMC", "FS", "FS|CMC"], leave=True)
    # pbar_outer = tqdm(["FS", "FS|CMC"], leave=True)
    for feature_regex in pbar_outer:
        pbar_outer.set_description(feature_regex)
        features = df.filter(regex=feature_regex)
        cols = df.filter(regex="TARGET").columns.to_list()
        pbar = tqdm(cols, leave=True)
        for target in pbar:
            y = df[target]
            seq = StepUpSelect(
                n_features_to_select=100,
                tol=1e-5,
                estimator=LR(),
                direction="forward",
                scoring="explained_variance",
                cv=5,
                n_jobs=-1,
            )
            seq.fit(features.to_numpy(), y)
            idx = np.array(
                seq.get_feature_names_out(np.arange(features.shape[1]))
            ).astype(np.int64)
            s = np.array(seq.iteration_scores)
            best = 100 * np.max(seq.iteration_scores)
            n_best = np.min(np.where(s == s.max()))
            all_scores.append(
                DataFrame(
                    {
                        "Exp.Var": best,
                        "n_best": n_best,
                        "source": feature_regex,
                        "target": str(target).replace("TARGET__", ""),
                        "features": str(seq.iteration_features),
                    },
                    index=[count],
                )
            )
            name = str(target).replace("TARGET__", "")
            pbar.set_description(f"{name}: {np.round(best, 3)}% @ {n_best} features")
            count += 1
    scores = pd.concat(all_scores, axis=0)
    scores.to_parquet(scores_out)
    print(f"Saved scores to {scores_out}")
    return scores


if __name__ == "__main__":
    reduce_cmc = False
    reduce_targets = True
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )

    scores = stepup_feature_select()
    scores.insert(3, "CMC ratio", 0.0)
    for i in range(scores.shape[0]):
        regex = scores.loc[i, "source"]
        if regex == "CMC":
            continue
        n = int(scores.loc[i, "n_best"])  # type: ignore
        idx = np.array(eval(scores.loc[i, "features"]), dtype=np.int64)
        cols = df.filter(regex=regex).columns.to_numpy()[idx]
        n_cmc = sum("CMC" in col for col in cols)
        scores.loc[i, "CMC ratio"] = n_cmc / n
    print(scores)
    pivoted = (
        scores.drop(columns="features")
        .groupby("source")
        .apply(
            lambda g: g.drop(columns="source").sort_values(
                by=["Exp.Var", "target"], ascending=[False, True]
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
