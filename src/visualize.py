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
from math import ceil, floor, sqrt
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
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import CACHED_RESULTS, PLOTS, RESULTS, TABLES
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus
from src.munging.fs_stats import load_HCP_complete, load_phenotypic_data


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


def _correlate(feats_targs_col: tuple[DataFrame, DataFrame, str]) -> DataFrame:
    features, targets, col = feats_targs_col
    corrs = (
        features.corrwith(targets[col], method="spearman", drop=False).to_frame().T.copy()
    )
    # pandas throwing in garbage here for some reason
    # corrs = corrs.filter(regex="CMC").copy()
    missing = list(set(features.columns).difference(corrs.columns))
    n = len(missing)
    pad = DataFrame(data=np.full([1, n], np.nan), columns=missing)
    corrs = pd.concat([corrs, pad], axis=1)
    # for c in missing:
    #     corrs.loc[:, c] = np.nan
    corrs.index = [str(col).replace("target__", "")]  # type: ignore
    return corrs


def compute_HCP_correlations(
    pheno: PhenotypicFocus, cmc_only: bool = True, use_cache: bool = True
) -> DataFrame:
    c = "cmc" if cmc_only else "all"
    out = CACHED_RESULTS / f"HCP_{c}_corrs_{pheno.value}.parquet"
    if out.exists() and use_cache:
        return pd.read_parquet(out)

    df = load_HCP_complete(focus=pheno)
    targets = df.filter(regex="target__").copy()
    targets = targets.dropna(axis=1, how="all")
    means = targets.mean()
    targets.fillna(means, inplace=True)
    const = targets.std(axis=0) == 0
    targets = targets.loc[:, ~const]

    features = df.filter(regex="CMC").copy() if cmc_only else df.copy()
    features = features.select_dtypes(exclude="O")
    sds = features.std(axis=0)
    const = (sds == 0) | (sds.isna())
    features = features.loc[:, ~const]

    args = [(features, targets, col) for col in sorted(targets.columns)]
    dfs = process_map(_correlate, args, total=len(args))
    # dfs = []
    # for col in tqdm(sorted(targets.columns), total=targets.shape[1]):
    #     corrs = (
    #         features.corrwith(targets[col], method="spearman", drop=False)
    #         .to_frame()
    #         .T.copy()
    #     )
    #     # pandas throwing in garbage here for some reason
    #     corrs = corrs.filter(regex="CMC").copy()
    #     missing = list(set(features.columns).difference(corrs.columns))
    #     n = len(missing)
    #     pad = DataFrame(data=np.full([1, n], np.nan), columns=missing)
    #     corrs = pd.concat([corrs, pad], axis=1)
    #     # for c in missing:
    #     #     corrs.loc[:, c] = np.nan
    #     corrs.index = [str(col).replace("target__", "")]  # type: ignore
    #     dfs.append(corrs)
    # # rows are targets, columns are features
    corrs = pd.concat(dfs, axis=0)
    corrs.to_parquet(out)
    print(f"Saved correlations to {out}")
    return corrs


def sort_correlations(corrs: DataFrame) -> DataFrame:
    # corrs = corrs.copy()[~corrs.index.str.contains("fs_")]
    tall = (
        corrs.stack()  # type: ignore
        .reset_index(name="r_spearman")
        .rename(columns={"index": "target", "level_1": "feature"})
    )
    tall["r_abs"] = tall["r_spearman"].abs()
    # tall = tall[tall["r_abs"] > 0.1]
    tall = tall.sort_values(by="r_abs", ascending=False).drop(columns="r_abs")
    key = pd.read_csv(focus.hcp_dict_file())
    key.rename(columns={"columnHeader": "target"}, inplace=True)
    key.rename(columns=str.lower, inplace=True)
    key["target"] = key["target"].str.lower()
    key = key.loc[:, ["target", "fulldisplayname", "description"]]

    tall["feature"] = tall["feature"].str.replace("target__", "")
    tall = pd.merge(tall, key, how="left", on="target")
    tall.rename(
        columns={"fulldisplayname": "target_displayname", "description": "target_desc"},
        inplace=True,
    )
    key.rename(columns=dict(target="feature"), inplace=True)
    tall = pd.merge(tall, key, how="left", on="feature")
    tall.rename(
        columns={
            "fulldisplayname": "feature_displayname",
            "description": "feature_desc",
        },
        inplace=True,
    )

    tall = tall.loc[
        :,
        [
            "r_spearman",
            "feature",
            "target",
            "target_displayname",
            "feature_displayname",
            "target_desc",
            "feature_desc",
        ],
    ]
    # remove meaningless correlations
    tall = tall[~tall["target"].str.contains("fs_")]
    tall = tall[tall["target"] != tall["feature"]]
    tall = tall[tall["r_spearman"].abs() < (1.0 - 1e-5)]

    tall["r_abs"] = tall["r_spearman"].abs()
    tall.insert(2, "feat_is_neuro", tall["feature"].str.contains("fs_|CMC"))
    tall = tall.sort_values(by=["feat_is_neuro", "r_abs"], ascending=[False, False])

    return tall


def make_cmc_v_freesurfer_histogram() -> None:
    ax: Axes
    fig, axes = plt.subplots(ncols=3, sharex=True)

    phenos = [PhenotypicFocus.All, PhenotypicFocus.Reduced, PhenotypicFocus.Focused]
    for i, (ax, pheno) in enumerate(zip(axes, phenos)):
        corrs = compute_HCP_correlations(pheno=pheno, cmc_only=False, use_cache=True)
        tall = sort_correlations(corrs)
        df = tall[tall["feat_is_neuro"]]
        fs = df[df.feature.str.contains("fs_")]
        cmc = df[df.feature.str.contains("CMC")]
        lab = lambda s: s if i == 2 else None

        cargs = dict(color="#ff6505", alpha=0.3)
        ax.hist(
            cmc["r_spearman"], bins=100, density=False, label=lab("CMC Metrics"), **cargs
        )
        cargs = dict(color="#056dff", alpha=0.3)
        ax.hist(
            fs["r_spearman"],
            bins=100,
            density=False,
            label=lab("FreeSurfer Metrics"),
            **cargs,
        )

        desc = f"{pheno.desc().capitalize()} Targets"
        ax.set_xlabel("Correlation (Spearman)")
        ax.set_ylabel("Number of Correlations")
        ax.set_title(desc)
        ymax = ax.get_ylim()[1]
        ax.vlines(
            x=cmc["r_spearman"].mean(),
            ymin=0,
            ymax=ymax,
            color="#ff6505",
            label=lab("CMC mean"),
        )
        ax.vlines(
            x=fs["r_spearman"].mean(),
            ymin=0,
            ymax=ymax,
            color="#056dff",
            label=lab("FreeSurfer mean"),
        )
        ax.set_xlim(-0.25, 0.3)
        ax.set_ylim(0, ymax)
        if i == 2:
            ax.legend().set_visible(True)

    fig.suptitle("Distribution of Univariate Metric Correlations with Phenotypic Targets")
    fig.set_size_inches(w=14, h=6)
    fig.tight_layout()
    out = str(PLOTS / f"cmc_v_freesurfer_correlation_distributions.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved correlations histogram to {out}")


if __name__ == "__main__":
    focus = pheno = PhenotypicFocus.All
    cmc_only = False
    c = "CMC-features" if cmc_only else "all-features"

    # pheno = load_phenotypic_data(FreesurferStatsDataset.HCP, pheno=focus)
    # tall = sort_correlations(corrs)

    # make_cmc_v_freesurfer_histogram()

    all_pheno = PhenotypicFocus.All
    df = load_HCP_complete(focus=all_pheno)
    df_out = TABLES / f"HCP_{c}_{all_pheno.value}-targets_data.csv.gz"
    df.to_csv(
        df_out,
        compression={"method": "gzip", "compresslevel": 9},
    )
    print(f"Saved raw data table to {df_out}")

    corrs = compute_HCP_correlations(pheno=all_pheno, cmc_only=False, use_cache=True)
    corrs_out = TABLES / f"HCP_{c}_{all_pheno.value}-targets_raw_correlations.csv.gz"
    corrs.to_csv(
        corrs_out,
        compression={"method": "gzip", "compresslevel": 9},
    )
    print(f"Saved raw correlations to {corrs_out}")

    for pheno in PhenotypicFocus:
        corrs = compute_HCP_correlations(pheno=focus, cmc_only=False, use_cache=True)
        tall = sort_correlations(corrs)
        tall_out = (
            TABLES
            / f"HCP_{c}_{pheno.value}-targets_sorted_correlations_with_descriptions.csv.gz"
        )
        tall.to_csv(
            tall_out,
            compression={"method": "gzip", "compresslevel": 9},
        )
        print(f"Saved sorted correlations to {tall_out}")

    sys.exit()

    sbn.clustermap(corrs, cmap="vlag", xticklabels=1, yticklabels=1)
    fig = plt.gcf()
    fig.set_size_inches(w=60, h=30)
    fig.tight_layout()
    cluster_out = str(PLOTS / f"HCP_{focus.value}_clustermap.png")
    fig.savefig(cluster_out, dpi=300)
    plt.close()
    print(f"Saved cluster map to {cluster_out}")

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
