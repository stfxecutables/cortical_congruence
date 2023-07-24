from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from seaborn import FacetGrid
from typing_extensions import Literal

from src.analysis.feature_select.stepwise import (
    means_to_best_n_feats,
    nested_results_to_means,
    nested_stepup_feature_select,
)
from src.constants import PLOTS
from src.enumerables import FeatureRegex, FreesurferStatsDataset, RegressionMetric

# matplotlib.use("QtAgg")


TARGET_PALETTE = {
    # "mae": "#f2830d",
    "smae": "#0d7ff2",
    "exp-var": "#000000",
    # "test_mae": "#f2830d",
    "test_smae": "#0d7ff2",
    "test_exp-var": "#000000",
}
METRIC_LINE_STYLES = {  # first element of tuple is segment width, second element is space
    # "mae": (1, 1),  # densely dotted
    "smae": (1, 1),
    "exp-var": (1, 1),
    # "test_mae": (1, 0),  # solid?
    "test_smae": (1, 0),
    "test_exp-var": (1, 0),
}
METRIC_MARKERS = {
    # "mae": "*",
    "smae": "*",
    "exp-var": "*",
    # "test_mae": ".",  # solid?
    "test_smae": ".",
    "test_exp-var": ".",
}
SOURCE_ORDER = ["FS", "CMC", "FS|CMC"]


def plot_k_curves(info: DataFrame, average: bool = True) -> None:
    means = nested_results_to_means(info)
    means = means.drop(
        columns=[
            "r2",
            "test_r2",
            "mad",
            "smad",
            "test_mad",
            "test_smad",
            "mae",
            "test_mae",
        ]
    )
    means["target"] = means["target"].str.replace("REG__", "")
    if average:
        k_means = (
            means.groupby(["source", "model", "target", "selection_iter"])
            .mean(numeric_only=True)
            .drop(columns="outer_fold")
            .reset_index()
        )
        df = k_means
    else:
        df = means.drop(columns=["outer_fold", "features"])
    df = df.melt(id_vars=df.columns.to_list()[:4], var_name="metric", value_name="value")

    df["is_test"] = df["metric"].str.contains("test")
    styles = METRIC_LINE_STYLES
    grid: FacetGrid = sbn.relplot(
        kind="line",
        data=df,
        x="selection_iter",
        y="value",
        hue="metric",
        palette=TARGET_PALETTE,
        hue_order=TARGET_PALETTE.keys(),
        col="source",
        col_order=SOURCE_ORDER,
        row="target",
        style="metric",
        style_order=styles.keys(),
        dashes=list(styles.values()),
        # markers=None if average else METRIC_MARKERS,
        # legend=False,
        height=2,
        aspect=1,  # width = height * aspect
        # errorbar=("ci", 99.99),
        errorbar=("pi", 100),
        # n_boot=100,  # don't need many for max
        n_boot=5,  # don't need many for max
    )
    grid.set_titles("{row_name} ({col_name} Features)")
    grid.set_ylabels("")
    grid.set_xlabels("N Selected Features")
    sbn.move_legend(grid, loc=(0.05, 0.95))
    ax: Axes
    xmin, xmax = np.percentile(df["selection_iter"], [0, 100])
    for ax in grid.axes.flat:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ax.get_ylim())
        ax.hlines(
            y=[0, 1],
            xmin=xmin,
            xmax=xmax,
            colors="red",
            alpha=0.7,
            lw=0.8,
        )
        ax.fill_between(
            x=[xmin, xmax],
            y1=[-2.0, -2.0],
            y2=[0.0, 0.0],
            color="#f2830d",
            alpha=0.1,
        )
        ax.fill_between(
            x=[xmin, xmax],
            y1=[1.0, 1.0],
            y2=[5.0, 5.0],
            color="#f2830d",
            alpha=0.1,
        )
    out = PLOTS / f"hcp_k_curves_{'mean' if average else ''}.png"
    fig: Figure
    fig = grid.figure
    fig.set_size_inches(w=10, h=30)
    fig.tight_layout()
    fig.savefig(str(out), dpi=300)
    print(f"Saved plot to {out}")
    plt.close()


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
