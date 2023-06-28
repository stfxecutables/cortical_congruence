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

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil, sqrt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal
import matplotlib
import seaborn as sbn
from pandas import Series


from src.munging.fs_stats import load_HCP_complete
from src.constants import PLOTS, CACHED_RESULTS


def best_rect(m: int) -> tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(floor(sqrt(m)))
    high = int(ceil(sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for prod in prods:
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def visualize_HCP_targets() -> None:
    matplotlib.use("QtAgg")
    df = load_HCP_complete()
    targets = df.filter(regex="target__")
    nrows, ncols = best_rect(targets.shape[1])
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    fig.set_size_inches(w=20, h=15)
    ax: Axes
    for col, ax in zip(sorted(targets.columns), axes.flat):
        ax.hist(targets[col], color="black", bins=50, density=True, alpha=0.7)
        sbn.kdeplot(x=targets[col], color="black", ax=ax)
        ax.set_title(col.replace("target__", ""))
        ax.set_xlabel("")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    out = CACHED_RESULTS / "HCP_corrs.parquet"
    if out.exists():
        corrs = pd.read_parquet(out)

    # df = load_HCP_complete(focused_pheno=False)
    # targets = df.filter(regex="target__").copy()
    # targets = targets.dropna()
    # const = targets.std(axis=0) == 0
    # targets = targets.loc[:, ~const]
    # cmc = df.filter(regex="CMC").copy()
    # row_names, dfs = [], []
    # for i, col in tqdm(enumerate(sorted(targets.columns)), total=targets.shape[1]):
    #     corrs = (
    #         cmc.corrwith(targets[col], method="spearman", drop=False).to_frame().T.copy()
    #     )
    #     # pandas throwing in garbage here for some reason
    #     corrs = corrs.filter(regex="CMC").copy()
    #     missing = set(cmc.columns).difference(corrs.columns)
    #     for c in missing:
    #         corrs[c] = np.nan
    #     corrs.index = [str(col).replace("target__", "")]

    #     dfs.append(corrs)

    # rows are targets, columns are features
    # corrs = pd.concat(dfs, axis=0)
    if not out.exists():
        corrs.to_parquet(out)
    corrs = corrs[~corrs.index.str.contains("fs_")]

    sbn.clustermap(corrs, cmap="vlag", xticklabels=1, yticklabels=1)
    fig = plt.gcf()
    fig.set_size_inches(w=60, h=30)
    fig.tight_layout()
    fig.savefig(PLOTS / "HCP_clustermap.png", dpi=300)
    plt.close()
    print("Saved cluster map")

    # mean_corrs = corrs.mean(axis=0)
    # idx = mean_corrs.abs().sort_values(ascending=False).index
    # sbn.heatmap(
    #     corrs[idx],
    #     cmap="vlag",
    #     square=True,
    #     xticklabels=1,
    #     yticklabels=1,
    #     cbar_kws={"shrink": 0.5},
    # )
    # fig = plt.gcf()
    # fig.set_size_inches(w=60, h=40)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9, bottom=0.1)
    # fig.savefig(PLOTS / "HCP_heatmap_feature-sorted.png", dpi=300)
    # plt.close()
    # print("Saved feature-sorted heatmap")

    # mean_corrs = corrs.mean(axis=1)
    # idx = mean_corrs.abs().sort_values(ascending=True).index
    # sbn.heatmap(
    #     corrs.loc[idx],
    #     cmap="vlag",
    #     square=True,
    #     xticklabels=1,
    #     yticklabels=1,
    #     cbar_kws={"shrink": 0.5},
    # )
    # fig = plt.gcf()
    # fig.set_size_inches(w=60, h=40)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.9, bottom=0.1)
    # fig.savefig(PLOTS / "HCP_heatmap_target-sorted.png", dpi=300)
    # plt.close()
    # print("Saved target-sorted heatmap")
