from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from src.enumerables import (
    ClassificationModel,
    FeatureRegex,
    FreesurferStatsDataset,
    RegressionModel,
)
from src.munging.hcp import PhenotypicFocus


def load_preprocessed(
    dataset: FreesurferStatsDataset,
    regex: FeatureRegex,
    models: RegressionModel
    | ClassificationModel
    | tuple[RegressionModel, ClassificationModel],
) -> DataFrame:
    df = dataset.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )
    feats = df.filter(regex=regex.value)
    feature_cols = feats.columns
    df.loc[:, feature_cols] = feats.fillna(feats.mean())

    if isinstance(models, tuple):
        reg_model, cls_model = models
    elif isinstance(models, RegressionModel):
        reg_model, cls_model = models, None
    elif isinstance(models, ClassificationModel):
        reg_model, cls_model = None, models
    else:
        reg_model = cls_model = models

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
