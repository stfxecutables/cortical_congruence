from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import pandas as pd

from src.analysis.feature_select.stepwise import (
    means_to_best_n_feats,
    nested_results_to_means,
    nested_stepup_feature_select,
)
from src.enumerables import (
    ClassificationMetric,
    FeatureRegex,
    FreesurferStatsDataset,
    RegressionMetric,
)
from src.visualization.plotting import plot_regression_k_curves

if __name__ == "__main__":
    pd.options.display.max_rows = 500
    for data in [FreesurferStatsDataset.ABIDE_I, FreesurferStatsDataset.ABIDE_II]:
        infos = []
        for regex in FeatureRegex:
            infos.append(
                nested_stepup_feature_select(
                    dataset=data,
                    feature_regex=regex,
                    max_n_features=50,
                    bin_stratify=True,
                    inner_progress=True,
                    use_cached=True,
                    load_complete=True,
                )
            )
        folds = pd.concat(infos, axis=0, ignore_index=True)
        plot_regression_k_curves(data=data, info=folds, average=False)
    sys.exit()
    means = nested_results_to_means(folds)
    rm = RegressionMetric.ExplainedVariance
    cm = ClassificationMetric.F1
    bests_r, bests_c = means_to_best_n_feats(means, metric=(rm, cm), test=False)
    print(folds)
