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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

from src.analysis.feature_select.stepwise import (
    means_to_best_n_feats,
    nested_results_to_means,
    nested_stepup_feature_select,
)
from src.enumerables import FeatureRegex, FreesurferStatsDataset, RegressionMetric

TARGET_PALETTE = {
    "mae": "#0d7ff2",
    "smae": "#f2830d",
    "exp-var": "#000000",
    "test_mae": "#0d7ff2",
    "test_smae": "#f2830d",
    "test_exp-var": "#000000",
}
METRIC_STYLES = {  # first element of tuple is segment width, second element is space
    "mae": (1, 1),  # densely dotted
    "smae": (1, 1),
    "exp-var": (1, 1),
    "test_mae": (1, 0),  # solid?
    "test_smae": (1, 0),
    "test_exp-var": (1, 0),
}
SOURCE_ORDER = ["FS", "CMC", "FS|CMC"]


def plot_k_curves(info: DataFrame) -> None:
    means = nested_results_to_means(info)
    means = means.drop(columns=["r2", "test_r2", "mad", "smad", "test_mad", "test_smad"])
    means["target"] = means["target"].str.replace("REG__", "")
    k_means = (
        means.groupby(["source", "model", "target", "selection_iter"])
        .mean(numeric_only=True)
        .drop(columns="outer_fold")
        .reset_index()
    )
    df = k_means
    df = df.melt(id_vars=df.columns.to_list()[:4], var_name="metric", value_name="value")
    df["is_test"] = df["metric"].str.contains("test")
    sbn.relplot(
        kind="line",
        data=df,
        x="selection_iter",
        y="value",
        hue="metric",
        palette=TARGET_PALETTE,
        hue_order=TARGET_PALETTE.keys(),
        row="source",
        row_order=SOURCE_ORDER,
        col="target",
        style="metric",
        style_order=METRIC_STYLES.keys(),
        dashes=list(METRIC_STYLES.values()),
        # legend=False,
        height=2,
        aspect=2,  # width = height * aspect
    )
    plt.show()


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
            )
        )
    info = pd.concat(infos, axis=0, ignore_index=True)
    # plot_k_curves(info)
    # sys.exit()

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
