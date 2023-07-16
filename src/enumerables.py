from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import Enum
from typing import Any, Callable, Mapping, Union

import numpy as np
from lightgbm import LGBMRegressor as LGB
from numpy import ndarray
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import (
    LogisticRegressionCV,
    Ridge,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.svm import SVC, SVR

from src.constants import DATA
from src.constants import REGULARIZATION_ALPHAS as ALPHAS
from src.metrics import (
    acc_bal_scorer,
    acc_scorer,
    accuracy_score,
    auroc_scorer,
    balanced_accuracy_score,
    expl_var,
    expl_var_scorer,
    f1_score,
    f1_scorer,
    mad,
    mad_scorer,
    mae,
    mae_scorer,
    precision_score,
    precision_scorer,
    r2,
    r2_scorer,
    recall_score,
    recall_scorer,
    roc_auc_score,
    smad,
    smad_scorer,
    smae,
    smae_scorer,
)
from src.munging.remove_nans import remove_nans

Regressor = Union[LGB, SVR, LR, Dummy]
Metric = Callable[[ndarray, ndarray], float]


class Tag(Enum):
    Id = "ID"
    Info = "INFO"
    # Feature types
    Demographics = "DEMO"
    CMC = "CMC"
    FreeSurfer = "FS"
    Feature = "FEAT"
    # Target types
    Target = "TARGET"
    Regression = "REG"
    Classification = "CLS"
    Multiclass = "MULTI"
    # Feature or target construction types
    Reduced = "REDUCED"
    Constructed = "CONSTR"

    @staticmethod
    def combine(tags: list[Tag]) -> str:
        if len(tags) == 0:
            return ""

        if Tag.Id in tags and len(tags) > 1:
            raise ValueError("'Id' columns can only be tagged as 'Id'")
        if Tag.Info in tags and len(tags) > 1:
            raise ValueError("'Info' columns can only be tagged as 'Info'")

        if Tag.Feature in tags:
            if Tag.Id in tags:
                raise ValueError("Cannot tag column as both 'Id' and 'Feature'")
            if Tag.Info in tags:
                raise ValueError("Cannot tag column as both 'Info' and 'Feature'")

        if Tag.Target in tags:
            if Tag.Feature in tags:
                raise ValueError("Cannot tag column as both 'Feature' and 'Target'")
            if Tag.CMC in tags:
                raise ValueError("Cannot tag column as both 'CMC' and 'Target'")
            if Tag.Demographics in tags:
                raise ValueError("Cannot tag column as both 'Demographic' and 'Target'")
            if Tag.FreeSurfer in tags:
                raise ValueError("Cannot tag column as both 'FreeSurfer' and 'Target'")

        if (Tag.Reduced in tags) and (Tag.Constructed in tags):
            raise ValueError("Column can not be both 'Reduced' and 'Constructed'")

        combined = []
        for tag in Tag:
            if tag in tags:
                combined.append(tag.value)
        return "__".join(combined)


class ROISource(Enum):
    Base = "base"
    Pial = "pial"
    DKT = "dkt"


class FreesurferStatsDataset(Enum):
    ABIDE_I = "abide-i"
    ABIDE_II = "abide-ii"
    ADHD_200 = "adhd-200"
    HBN = "hbn"
    HCP = "hcp"
    QTAB = "qtab"  # special, since two sessions
    QTIM = "qtim"  # special, since two sessions

    def root(self) -> Path:
        return {
            FreesurferStatsDataset.ABIDE_I: DATA / "ABIDE-I",
            FreesurferStatsDataset.ABIDE_II: DATA / "ABIDE-II",
            FreesurferStatsDataset.ADHD_200: DATA / "ADHD-200",
            FreesurferStatsDataset.HBN: DATA / "HBN",
            FreesurferStatsDataset.HCP: DATA / "HCP",
            FreesurferStatsDataset.QTAB: DATA / "QTAB",
            FreesurferStatsDataset.QTIM: DATA / "QTIM",
        }[self]

    def phenotypic_file(self) -> Path:
        root = self.root() / "phenotypic_data"

        return {
            FreesurferStatsDataset.ABIDE_I: root / "ABIDE_I_Phenotypic_V1_0b.csv",
            FreesurferStatsDataset.ABIDE_II: root / "ABIDEII_Composite_Phenotypic.csv",
            FreesurferStatsDataset.ADHD_200: root
            / "ADHD200_preprocessed_phenotypics.tsv",
            FreesurferStatsDataset.HBN: root / "HBN_R10_Pheno.csv",
            FreesurferStatsDataset.HCP: root / "unrestricted_behavioral.csv",
            FreesurferStatsDataset.QTAB: root / "phenotypic_data.tsv",
            FreesurferStatsDataset.QTIM: root / "participants.tsv",
        }[self]

    def load_pheno(self, focus: Any | None = None) -> DataFrame:
        """Groteqsque hackery to have a clean interface from the enum..."""
        from src.munging.abide import (
            load_abide_i_pheno,
            load_abide_ii_pheno,
            load_adhd200_pheno,
        )
        from src.munging.hcp import PhenotypicFocus, load_hcp_phenotypic

        if focus is None:
            focus = PhenotypicFocus.All

        assert isinstance(focus, PhenotypicFocus)

        def not_implemented() -> None:
            raise NotImplementedError()

        return {
            FreesurferStatsDataset.ABIDE_I: load_abide_i_pheno,
            FreesurferStatsDataset.ABIDE_II: load_abide_ii_pheno,
            FreesurferStatsDataset.ADHD_200: load_adhd200_pheno,
            FreesurferStatsDataset.HBN: lambda: not_implemented(),
            FreesurferStatsDataset.HCP: lambda: load_hcp_phenotypic(focus),
            FreesurferStatsDataset.QTAB: lambda: not_implemented(),
            FreesurferStatsDataset.QTIM: lambda: not_implemented(),
        }[self]()

    def load_complete(
        self,
        focus: Any | None = None,
        reduce_targets: bool = True,
        reduce_cmc: bool = False,
    ) -> DataFrame:
        from src.munging.abide import (
            load_abide_i_complete,
            load_abide_ii_complete,
            load_adhd200_complete,
        )
        from src.munging.hcp import PhenotypicFocus, load_HCP_complete

        if focus is None:
            focus = PhenotypicFocus.All

        assert isinstance(focus, PhenotypicFocus)

        def not_implemented() -> None:
            raise NotImplementedError()

        df = {
            FreesurferStatsDataset.ABIDE_I: load_abide_i_complete,
            FreesurferStatsDataset.ABIDE_II: load_abide_ii_complete,
            FreesurferStatsDataset.ADHD_200: load_adhd200_complete,
            FreesurferStatsDataset.HBN: lambda: not_implemented(),
            FreesurferStatsDataset.QTAB: lambda: not_implemented(),
            FreesurferStatsDataset.QTIM: lambda: not_implemented(),
            FreesurferStatsDataset.HCP: lambda: load_HCP_complete(
                focus=focus, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
            ),
        }[self]()
        return remove_nans(df)

    def data_dictionary(self) -> Path:
        root = self.root() / "phenotypic_data"

        return {
            FreesurferStatsDataset.ABIDE_I: root / "ABIDE_I_Phenotypic_V1_0b.csv",
            FreesurferStatsDataset.ABIDE_II: root / "ABIDEII_Composite_Phenotypic.csv",
            FreesurferStatsDataset.ADHD_200: root
            / "ADHD200_preprocessed_phenotypics.tsv",
            FreesurferStatsDataset.HBN: root / "HBN_R10_Pheno.csv",
            FreesurferStatsDataset.HCP: root / "unrestricted_behavioral.csv",
            FreesurferStatsDataset.QTAB: root / "phenotypic_data.tsv",
            FreesurferStatsDataset.QTIM: root / "participants.tsv",
        }[self]

    def freesurfer_files(self, ses2: bool = False) -> tuple[Path, Path] | Path | None:
        root = self.root()

        file = {
            FreesurferStatsDataset.ABIDE_I: None,
            FreesurferStatsDataset.ABIDE_II: None,
            FreesurferStatsDataset.ADHD_200: None,
            FreesurferStatsDataset.HBN: None,
            FreesurferStatsDataset.HCP: root / "unrestricted_hcp_freesurfer.csv",
            FreesurferStatsDataset.QTAB: None,
            FreesurferStatsDataset.QTIM: (
                (
                    root / "stats/fs5.3_ses-01.tsv",
                    root / "stats/fs5.3_subfields6_ses-01.tsv",
                ),
                (
                    root / "stats/fs5.3_ses-02.tsv",
                    root / "stats/fs5.3_subfields6_ses-02.tsv",
                ),
            ),
        }[self]
        if not isinstance(file, tuple):
            return file

        files = file[1] if ses2 else file[0]
        return files

    def has_pial_stats(self) -> bool:
        """
        Returns
        -------
        has_pial: bool
            Returns True if `*h.aparc.pial.stats` files exist for most subjects.

        """
        return {
            FreesurferStatsDataset.ABIDE_I: False,  # definitely not
            FreesurferStatsDataset.ABIDE_II: True,
            FreesurferStatsDataset.ADHD_200: True,
            FreesurferStatsDataset.HBN: True,
            FreesurferStatsDataset.HCP: False,  # does not seem to
            FreesurferStatsDataset.QTAB: False,  # not currently
            FreesurferStatsDataset.QTIM: False,  # not currently
        }[self]

    def has_dkt_atlas_stats(self) -> bool:
        """
        Returns
        -------
        has_stas: bool
            Returns True if `*h.aparc.DKTatlas.stats` files exist for most subjects.

        """
        return {
            FreesurferStatsDataset.ABIDE_I: False,  # definitely not
            FreesurferStatsDataset.ABIDE_II: True,
            FreesurferStatsDataset.ADHD_200: True,
            FreesurferStatsDataset.HBN: True,
            FreesurferStatsDataset.HCP: False,  # does not seem to
            FreesurferStatsDataset.QTAB: False,  # not currently
            FreesurferStatsDataset.QTIM: False,  # not currently
        }[self]

    def load_stats_table(self) -> DataFrame:
        """Depending on data structure, either loop over stats files and build
        stats tables for each subject, or convert HCP and other pre-made table
        data to a common form, described below

        Notes
        -----
        Common table structure is as follows:

        """
        # TODO: save table in cached_results
        if self in [
            FreesurferStatsDataset.ABIDE_I,
            FreesurferStatsDataset.ABIDE_II,
            FreesurferStatsDataset.ADHD_200,
        ]:
            root = self.root()
        raise NotImplementedError()


class RegressionModel(Enum):
    LightGBM = "lgb"
    Linear = "linear"
    SVR = "svr"
    Dummy = "dummy"
    MLP = "mlp"
    Ridge = "ridge"
    Lasso = "lasso"

    def get(self, params: Mapping | None = None) -> Regressor:
        if params is None:
            params = dict()
        lasso = dict(
            precompute=True,
            selection="random",
            max_iter=5000,
            alphas=ALPHAS,
            cv=3,
        )
        return {
            RegressionModel.LightGBM: lambda: LGB(**params),
            RegressionModel.Linear: lambda: LR(**params),
            RegressionModel.SVR: lambda: SVR(**params),
            RegressionModel.Dummy: lambda: Dummy(strategy="mean"),
            RegressionModel.MLP: lambda: MLP(**params),
            RegressionModel.Ridge: lambda: Ridge(**params),
            RegressionModel.Lasso: lambda: LassoCV(**{**params, **lasso}),
        }[self]()


class ClassificationModel(Enum):
    Logistic = "logistic"
    Ridge = "ridge"
    SGD = "sgd"
    SVC = "svc"
    Dummy = "dummy"

    def get(self, params: Mapping | None = None) -> Regressor:
        if params is None:
            params = dict()
        logistic = dict(
            Cs=np.sort(np.array([1 / (2 * alpha) for alpha in ALPHAS])),
            cv=3,
            penalty="l1",
            solver="saga",  # needed for "l1" and multinomial
            multi_class="multinomial",
            max_iter=500,
        )
        ridge = dict(alphas=np.logspace(start=-5, stop=5, num=100, base=10), cv=3)
        sgd = dict(
            loss="log_loss",
            penalty="l1",
            alpha=1e-4,
            n_iter_no_change=5,
        )
        return {
            ClassificationModel.Logistic: lambda: LogisticRegressionCV(
                **{**params, **logistic}
            ),
            ClassificationModel.SGD: lambda: SGDClassifier(**{**params, **sgd}),
            ClassificationModel.Ridge: lambda: RidgeClassifierCV(**{**params, **ridge}),
            ClassificationModel.SVC: lambda: SVC(**params),
            ClassificationModel.Dummy: lambda: DummyClassifier(strategy="most_frequent"),
        }[self]()


class RegressionMetric(Enum):
    MeanAbsoluteError = "mae"
    ScaledMeanAbsoluteError = "smae"
    MedianAbsoluteDeviation = "mad"
    ScaledMedianAbsoluteDeviation = "smad"
    ExplainedVariance = "exp-var"
    RSquared = "r2"

    def scorer(self) -> Metric:
        metrics: dict[RegressionMetric, Metric] = {
            RegressionMetric.MeanAbsoluteError: mae_scorer,
            RegressionMetric.ScaledMeanAbsoluteError: smae_scorer,
            RegressionMetric.MedianAbsoluteDeviation: mad_scorer,
            RegressionMetric.ScaledMedianAbsoluteDeviation: smad_scorer,
            RegressionMetric.ExplainedVariance: expl_var_scorer,
            RegressionMetric.RSquared: r2_scorer,
        }
        return metrics[self]

    @staticmethod
    def scorers() -> dict[str, Metric]:
        return {reg.value: reg.scorer() for reg in RegressionMetric}

    @staticmethod
    def no_proba_scorers() -> dict[str, Metric]:
        return RegressionMetric.scorers()

    @staticmethod
    def inverted() -> list[RegressionMetric]:
        return [
            RegressionMetric.MeanAbsoluteError,
            RegressionMetric.ScaledMeanAbsoluteError,
            RegressionMetric.MedianAbsoluteDeviation,
            RegressionMetric.ScaledMedianAbsoluteDeviation,
        ]

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        metric = {
            RegressionMetric.MeanAbsoluteError: mae,
            RegressionMetric.ScaledMeanAbsoluteError: smae,
            RegressionMetric.MedianAbsoluteDeviation: mad,
            RegressionMetric.ScaledMedianAbsoluteDeviation: smad,
            RegressionMetric.ExplainedVariance: expl_var,
            RegressionMetric.RSquared: r2,
        }[self]
        return metric(y_true, y_pred)


class ClassificationMetric(Enum):
    Accuracy = "acc"
    AUROC = "auroc"
    BalancedAccuracy = "acc_bal"
    F1 = "f1"
    Precision = "precision"
    Recall = "recall"

    def scorer(self) -> Metric:
        metrics = {
            ClassificationMetric.Accuracy: acc_scorer,
            ClassificationMetric.AUROC: auroc_scorer,
            ClassificationMetric.BalancedAccuracy: acc_bal_scorer,
            ClassificationMetric.F1: f1_scorer,
            ClassificationMetric.Precision: precision_scorer,
            ClassificationMetric.Recall: recall_scorer,
        }
        return metrics[self]

    @staticmethod
    def scorers() -> dict[str, Metric]:
        return {reg.value: reg.scorer() for reg in ClassificationMetric}

    @staticmethod
    def no_proba_scorers() -> dict[str, Metric]:
        return {
            reg.value: reg.scorer()
            for reg in ClassificationMetric
            if reg is not ClassificationMetric.AUROC
        }

    def __call__(self, y_true: ndarray, y_pred: ndarray) -> float:
        metric = {
            ClassificationMetric.Accuracy: accuracy_score,
            ClassificationMetric.AUROC: lambda y_true, y_pred: roc_auc_score(
                y_true, y_pred, average="macro", multi_class="ovr"
            ),
            ClassificationMetric.BalancedAccuracy: balanced_accuracy_score,
            ClassificationMetric.F1: lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="macro", zero_division=np.nan
            ),
            ClassificationMetric.Precision: lambda y_true, y_pred: precision_score(
                y_true, y_pred, average="macro", zero_division=np.nan
            ),
            ClassificationMetric.Recall: lambda y_true, y_pred: recall_score(
                y_true, y_pred, average="macro", zero_division=np.nan
            ),
        }[self]
        return metric(y_true, y_pred)


if __name__ == "__main__":
    # print(FreesurferStatsDataset.ABIDE_I.load_pheno())
    # print(FreesurferStatsDataset.ABIDE_II.load_pheno())
    # print(FreesurferStatsDataset.ADHD_200.load_pheno())
    # print(FreesurferStatsDataset.HCP.load_pheno())

    print(FreesurferStatsDataset.ABIDE_I.load_complete())
    print(FreesurferStatsDataset.ABIDE_II.load_complete())
    print(FreesurferStatsDataset.ADHD_200.load_complete())
    print(FreesurferStatsDataset.HCP.load_complete())
    # print(FreesurferStatsDataset.HBN.load_pheno())
