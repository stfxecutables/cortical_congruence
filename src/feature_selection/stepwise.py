from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
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
from sklearn.base import _fit_context  # type: ignore
from sklearn.base import BaseEstimator, clone, is_classifier
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
from sklearn.model_selection._split import BaseShuffleSplit, check_cv
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
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus, RegressionMetric
from src.munging.fs_stats import load_HCP_complete


@dataclass
class SelectArgs:
    estimator: Any
    X: ndarray
    y: ndarray
    mask: ndarray
    idx: ndarray
    direction: str
    scoring: RegressionMetric
    cv: Any


def _get_score(args: SelectArgs) -> tuple[float, DataFrame]:
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
    results = cross_validate(
        estimator,
        X_new,
        y,
        cv=cv,
        scoring=RegressionMetric.scorers(),
        n_jobs=1,
    )
    key = f"test_{scoring.value}"
    df = DataFrame(results).rename(columns=lambda s: s.replace("test_", ""))
    return float(results[key].mean()), df
    # return cross_val_score(
    #     estimator,
    #     X_new,
    #     y,
    #     cv=cv,
    #     scoring=scoring,
    #     n_jobs=1,
    # ).mean()


class StepwiseSelect(SequentialFeatureSelector):
    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        n_features_to_select: float | int | Literal["auto", "warn"] = "warn",
        tol: float | None = None,
        direction: Literal["forward", "backward"] = "forward",
        scoring: RegressionMetric = RegressionMetric.ExplainedVariance,
        cv: Iterable | int | BaseShuffleSplit | BaseCrossValidator = 5,
        inner_progress: bool = False,
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
        self.iteration_features: list[int] = []
        self.iteration_metrics: list[DataFrame] = []
        self.scoring = scoring
        self.inner_progress = inner_progress

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
        for _ in tqdm(range(n_iterations), leave=True, disable=not self.inner_progress):
            new_feature_idx, new_score, new_info = self._get_best_new_feature_score(
                cloned_estimator, X, y, cv, current_mask
            )
            self.iteration_scores.append(new_score)
            self.iteration_features.append(new_feature_idx)
            self.iteration_metrics.append(new_info)
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

        results: list[tuple[float, DataFrame]]
        results = Parallel(n_jobs=-1, verbose=0)(delayed(_get_score)(arg) for arg in args)

        scores: dict[int, tuple[float, DataFrame]] = {}
        for feature_idx, (score, info) in zip(candidate_feature_indices, results):
            scores[feature_idx] = (score, info)

        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx][0])
        best_score, best_info = scores[new_feature_idx]
        return new_feature_idx, best_score, best_info
