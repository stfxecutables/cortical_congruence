from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os

import pandas as pd

from src.analysis.feature_select.stepwise import (
    means_to_best_n_feats,
    nested_results_to_means,
    nested_stepup_feature_select,
)
from src.enumerables import FeatureRegex, FreesurferStatsDataset, RegressionMetric

if __name__ == "__main__":
    pd.options.display.max_rows = 500
    infos = []
    for regex in FeatureRegex:
        infos.append(
            nested_stepup_feature_select(
                dataset=FreesurferStatsDataset.HCP,
                feature_regex=regex,
                max_n_features=50,
                bin_stratify=True,
                inner_progress=True,
                use_outer_cached=True,
                use_inner_cached=True,
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
