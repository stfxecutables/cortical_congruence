from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from seaborn import FacetGrid

from src.analysis.feature_select.stepwise import (
    means_to_best_n_feats,
    nested_results_to_means,
    nested_stepup_feature_select,
)
from src.constants import PLOTS
from src.enumerables import FeatureRegex, FreesurferStatsDataset, RegressionMetric

# matplotlib.use("QtAgg")


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
                use_cached=True,
                load_complete=True,
            )
        )
    info = pd.concat(infos, axis=0, ignore_index=True)
    good = info[info.target.str.contains("int_g_like|language_perf")]
    good = good[good.inner_fold == 0]
    grid: FacetGrid
    grid = sbn.catplot(
        kind="count",
        x="selected",
        data=good[good.selection_iter <= 10],
        row="target",
        col="source",
    )
    grid.set_xlabels("Feature Index")
    grid.set_xticklabels([])
    grid.figure.savefig(PLOTS / "HCP_int_g_lang_perf_across_fold_counts_10.png")
    plt.close()

    # plot_k_curves(info, average=False)
    sys.exit()

    means = nested_results_to_means(info)
    means = means.drop(columns=["r2", "test_r2", "mad", "smad", "test_mad", "test_smad"])
    means["target"] = means["target"].str.replace("REG__", "")
    # mean across outer fold, for each k (k = n_selected_features)
    k_means = (
        means.groupby(["source", "model", "target", "selection_iter"])
        .mean(numeric_only=True)
        .drop(columns="outer_fold")
        .reset_index()
    )
    df = k_means
    df = df.melt(id_vars=df.columns.to_list()[:4], var_name="metric", value_name="value")
    df["is_test"] = df["metric"].str.contains("test")

    bests = means_to_best_n_feats(
        means, metric=RegressionMetric.ExplainedVariance, test=True
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
