from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import traceback
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral, Real
from typing import Any, Iterable, Literal, Protocol, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy import ndarray
from pandas import DataFrame, Index, Series
from sklearn.base import _fit_context  # type: ignore
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection._split import BaseShuffleSplit, check_cv
from tqdm import tqdm

from src.enumerables import ClassificationMetric, RegressionMetric


class Estimator(Protocol):
    def fit(self, X: DataFrame | ndarray, y: Series | ndarray) -> None:
        ...

    def predict(self, X: DataFrame | ndarray) -> Series | ndarray:
        ...


@dataclass
class SelectArgs:
    estimator: Any
    X_train: DataFrame | ndarray
    y_train: Series | ndarray
    mask: ndarray
    idx: ndarray
    scoring: Union[RegressionMetric, ClassificationMetric]
    cv: Any


class ForwardSelect:
    def __init__(
        self,
        estimator: BaseEstimator | Estimator,
        X_train: DataFrame | ndarray,
        y_train: Series | ndarray,
        X_test: DataFrame | ndarray,
        y_test: Series | ndarray,
        n_features_to_select: int = 50,
        scoring: Union[
            RegressionMetric, ClassificationMetric
        ] = RegressionMetric.ExplainedVariance,
        cv: Iterable | int | BaseShuffleSplit | BaseCrossValidator = 5,
        inner_progress: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_select = n_features_to_select
        self.n_features = X_train.shape[1]
        self.cv = check_cv(cv)
        self.n_jobs = n_jobs

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.iteration_features: list[int] = []
        self.iteration_scores: list[float] = []
        self.iteration_metrics: list[DataFrame] = []
        self.scoring = scoring
        self.inner_progress = inner_progress

        self.validate_args()

        self.mask = np.zeros(shape=self.n_features, dtype=bool)

    @cached_property
    def results(self) -> tuple[DataFrame, DataFrame]:
        if len(self.iteration_metrics) == 0:
            raise RuntimeError(
                f"Must call `{self.__class__.__name__}.select()` before returning results."
            )
        fold_infos, means = [], []
        for i, info in enumerate(self.iteration_metrics):
            fold_info = info.reset_index(names="inner_fold")
            fold_info.insert(0, "selected", self.iteration_features[i])
            fold_info.insert(0, "iter", i)

            fold_infos.append(fold_info)
            mean_info = info.mean().to_frame().T.copy()
            mean_info.insert(0, "selected", self.iteration_features[i])
            mean_info.insert(0, "iter", i)
            means.append(mean_info)

        fold_info = pd.concat(fold_infos, axis=0, ignore_index=True)
        mean_info = pd.concat(means, axis=0)
        needs_inversion = [metric.value for metric in RegressionMetric.inverted()]
        for colname in needs_inversion:
            fold_info.loc[:, colname] = -fold_info[colname].copy()
            mean_info.loc[:, colname] = -mean_info[colname].copy()

        mean_info["features"] = ""
        for i in range(len(self.iteration_features)):
            mean_info.loc[i, "features"] = str(self.iteration_features[: i + 1])
        return mean_info, fold_info

    def select(self) -> ForwardSelect:
        for _ in tqdm(range(self.n_select), leave=True, disable=not self.inner_progress):
            idx_selected, score, info = self._get_selected_feat_info()
            self.iteration_features.append(idx_selected)
            self.iteration_metrics.append(info)
            self.iteration_scores.append(score)
            self.mask[idx_selected] = True

        return self

    def _get_selected_feat_info(self) -> tuple[int, float, DataFrame]:
        """
        Returns
        -------
        selected_idx: int

        selected_score: float

        best_info: DataFrame
            DataFrame with 5 rows (if using 5-fold) and a column for each evaluation
            metric. Columns with "test_[metric_name]" are on the test set, and so
            row values are identical, whereas "[metric_name]" columns have a different
            value for each row (fold).

        """
        # get indices of nonzero mask locations, i.e. unchecked / unadded features
        estimator: Estimator = clone(self.estimator, safe=False)  # type: ignore
        remaining_idx = np.flatnonzero(~self.mask)
        args = [
            SelectArgs(
                estimator=estimator,
                X_train=self.X_train,
                y_train=self.y_train,
                cv=self.cv,
                mask=self.mask.copy(),
                idx=feature_idx,
                scoring=self.scoring,
            )
            for feature_idx in remaining_idx
        ]

        results: list[tuple[float, DataFrame]]
        results = Parallel(n_jobs=-1, verbose=0)(delayed(_get_score)(arg) for arg in args)  # type: ignore  # noqa

        scores: dict[int, tuple[float, DataFrame]] = {}
        for feature_idx, (score, info) in zip(remaining_idx, results):
            scores[feature_idx] = (score, info)

        selected_idx = max(scores, key=lambda feature_idx: scores[feature_idx][0])
        best_score, best_info = scores[selected_idx]

        X_test = self.X_test
        y_test = np.asarray(self.y_test)
        test: Estimator = clone(estimator, safe=False)  # type: ignore
        test.fit(self.X_train, self.y_train)
        y_pred = np.asarray(test.predict(X_test))
        for metric in self.scoring.__class__:
            best_info[f"test_{metric.value}"] = metric(y_test, y_pred)
        best_info.drop(columns=["fit_time", "score_time"], inplace=True)

        return selected_idx, best_score, best_info

    def validate_args(self) -> None:
        assert self.n_features == self.X_train.shape[1]
        assert self.X_train.shape[1] == self.X_test.shape[1]
        assert self.n_select < self.X_train.shape[1]


def _get_score(args: SelectArgs) -> tuple[float, DataFrame]:
    try:
        current_mask = args.mask
        feature_idx = args.idx
        estimator = args.estimator
        X_train = args.X_train
        y_train = args.y_train
        cv = args.cv
        scoring = args.scoring

        mask = current_mask.copy()
        mask[feature_idx] = True
        X_train = X_train[:, mask]
        scorers = scoring.__class__.scorers()

        results = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scorers,
            n_jobs=1,
            error_score="raise",
        )

        key = f"test_{scoring.value}"
        score = float(results[key].mean())
        df = DataFrame(results).rename(columns=lambda s: s.replace("test_", ""))
        return score, df
    except Exception as e:
        traceback.print_exc()
        raise e
