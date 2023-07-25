from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Index
from seaborn import FacetGrid

from src.analysis.dummy_perfs import get_dummy_perfs
from src.analysis.feature_select.stepwise import nested_results_to_means
from src.constants import (
    METRIC_LINE_STYLES,
    METRIC_PALETTE,
    PLOT_CLS_METRICS,
    PLOTS,
    SOURCE_ORDER,
)
from src.enumerables import FreesurferStatsDataset


def plot_regression_k_curves(
    data: FreesurferStatsDataset, info: DataFrame, average: bool = True
) -> None:
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
            "precision",
            "recall",
            "test_precision",
            "test_recall",
        ]
    )
    reg_idx = means["target"].str.contains("REG__")
    dummy_perfs = get_dummy_perfs()[1]
    dummy_perfs = dummy_perfs[dummy_perfs["data"] == data.value]
    means["target"] = means["target"].str.replace("REG__", "")
    means["target"] = means["target"].str.replace("CLS__", "")
    dummy_perfs["target"] = dummy_perfs["target"].str.replace("CLS__", "")
    dummy_perfs = dummy_perfs.drop(columns="data")
    dummy_perfs.index = Index(data=dummy_perfs["target"], name="target")
    dummy_perfs.drop(columns="target", inplace=True)
    reg_targets = means["target"][reg_idx].unique().tolist()
    cls_targets = means["target"][~reg_idx].unique().tolist()

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
        palette=METRIC_PALETTE,
        hue_order=METRIC_PALETTE.keys(),
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
        facet_kws=dict(sharey=False),
    )
    grid.set_titles("{row_name} ({col_name} Features)")
    grid.set_ylabels("")
    grid.set_xlabels("N Selected Features")
    sbn.move_legend(grid, loc=(0.05, 0.95))

    ax: Axes
    xmin, xmax = np.percentile(df["selection_iter"], [0, 100])
    for ax in grid.axes.flat:
        target = ax.get_title().split()[0]
        ax.set_xlim(xmin, xmax)
        if target in reg_targets:
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
        else:
            ax.set_ylim(0.1, 0.8)
            for metric in PLOT_CLS_METRICS:
                ax.hlines(
                    y=dummy_perfs.loc[target, metric],
                    xmin=xmin,
                    xmax=xmax,
                    colors=METRIC_PALETTE[metric],
                    lw=1.0,
                )

            ...
    out = PLOTS / f"{data.value}_k_curves{'_mean' if average else ''}.png"
    fig: Figure
    fig = grid.figure
    fig.set_size_inches(w=10, h=30)
    fig.tight_layout()
    fig.savefig(str(out), dpi=300)
    print(f"Saved plot to {out}")
    plt.close()
