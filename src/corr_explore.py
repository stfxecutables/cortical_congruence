from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Callable, cast
from pandas.errors import ParserError
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from src.constants import (
    ABIDE_II_ENCODING,
    ALL_STATSFILES,
    CACHED_RESULTS,
    DATA,
    HCP_FEATURE_INFO,
    MEMORY,
)
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import FactorAnalysis as FA


@MEMORY.cache
def reduce_HCP_clusters(data: DataFrame, clusters: list[DataFrame]) -> DataFrame:
    mappings = {
        "gambling_task_perc_larger": "gambling_perf",
        "mars_log_score": "mars",
        "language_task_median_rt": "language_rt",
        "psqi_score": "psqi_score",
        "relational_task_median_rt": "relational_rt",
        "emotion_task_median_rt": "emotion_rt",
        "pmat24_a_cr": "p_matrices",
        "wm_task_0bk_face_acc": "wm_face_acc",
        "social_task_median_rt_random": "social_rt",
        "social_task_perc_random": "social_random_perf",
        "language_task_acc": "language_perf",
        "social_task_perc_tom": "social_tom_perf",
        "gambling_task_median_rt_larger": "gambling_rt",
        "readeng_ageadj": "crystal_int",
        "wm_task_2bk_body_acc": "wm_perf",
        "fearaffect_unadj": "neg_emotionality",
        "wm_task_0bk_body_median_rt": "wm_rt",
    }
    reductions = []
    for cluster in tqdm(clusters, desc="Factor reducing..."):
        if len(cluster) > 0:
            name = mappings[str(cluster.iloc[0, 0])]
            feat_names = sorted(set(cluster.x.to_list() + cluster.y.to_list()))
            feats = data[feat_names]
            fa = FA(n_components=1, rotation="varimax")
            x = fa.fit_transform(feats.fillna(feats.median()))
            reductions.append(DataFrame(data=x, index=data["sid"], columns=[name]))
    return pd.concat(reductions, axis=1)


def get_cluster_corrs(
    corrs: DataFrame, min_cluster_size: int, epsilon: float = 0.2
) -> list[DataFrame]:
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=epsilon,
        metric="precomputed",
    )
    labels = hdb.fit_predict(1 - corrs.fillna(0.0).abs())
    labs = np.unique(labels)
    labs = labs[labs >= 0]  # remove noise labels
    sizes = []
    clusters = []
    for lab in labs:
        cluster = corrs.loc[labels == lab, labels == lab]
        idx = np.triu(np.ones(cluster.shape, dtype=bool), k=1)
        cluster = (
            cluster.where(idx)
            .stack()
            .reset_index()
            .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
        )
        sizes.append(len(cluster))
        cluster["abs"] = cluster["r"].abs()
        cluster = cluster.sort_values(by="abs", ascending=False).drop(columns="abs")
        clusters.append(cluster)
    clusters = sorted(clusters, key=lambda clust: len(clust))
    return clusters


def explore_corrs() -> None:
    df = pd.read_csv(FreesurferStatsDataset.HCP.phenotypic_file())
    extra = pd.read_csv(PhenotypicFocus.All.hcp_dict_file())
    df = df.rename(
        columns={"Subject": "sid", "Gender": "sex", "Age": "age_range"}
    ).rename(columns=str.lower)
    feats = extra["columnHeader"].str.lower()
    extra = extra.rename(columns=str.lower)
    extra["columnheader"] = extra["columnheader"].str.lower()

    available = list(set(feats).intersection(df.columns))
    freesurfer = sorted(filter(lambda f: "fs_" in f, available))
    remain = DataFrame(
        data=sorted(set(available).difference(freesurfer)), columns=["feature"]
    )
    feature_info = pd.read_csv(HCP_FEATURE_INFO)

    remain = pd.merge(
        remain,
        feature_info.loc[:, ["feature", "kind", "cardinality"]],
        how="left",
        on="feature",
    )
    remain.index = remain["feature"]  # type: ignore
    junk_features = remain.loc[remain["kind"].isin(["book_keeping", "sub_scale"])].index
    df = df.drop(columns=junk_features)
    remain = remain.drop(index=junk_features)

    df["age_range"] = df["age_range"].apply(
        lambda age: {"22-25": 0.0, "26-30": 1.0, "31-35": 2.0, "36+": 3.0}[age]
    )
    df["sex"] = df["sex"].apply(lambda s: 0 if s == "F" else 1).astype(np.float64)
    df.rename(columns={"age_range": "age_class"}, inplace=True)

    # Many remaining columns are extremely low quality and still useless due
    # to missing data or constancy. We remove those here.
    df = df.loc[:, df.isna().sum() < 200]  # removes 7 features
    df_orig = df.copy()
    df = df.drop(columns="sid")
    adj_drops = [
        "cardsort_unadj",
        "cogcrystalcomp_unadj",
        "cogearlycomp_unadj",
        "cogfluidcomp_unadj",
        "cogtotalcomp_unadj",
        "dexterity_unadj",
        "endurance_unadj",
        "flanker_unadj",
        "listsort_unadj",
        "odor_unadj",
        "picseq_unadj",
        "picvocab_unadj",
        "procspeed_unadj",
        "readeng_unadj",
        "strength_unadj",
        "taste_unadj",
    ]
    cluster_drops = [
        "gambling_task_perc_smaller",
        "gambling_task_reward_perc_smaller",
        "gambling_task_punish_perc_smaller",
        "scpt_fn",
        "scpt_tp",
        "scpt_tn",
        "scpt_fp",
    ]
    df = df.drop(columns=adj_drops + cluster_drops)

    corrs = df.corr()
    no_fs = df.drop(columns=df.filter(regex="fs_").columns)
    corrs = no_fs.corr(method="spearman")
    clusters = get_cluster_corrs(corrs, min_cluster_size=3, epsilon=0.2)
    reduced = reduce_HCP_clusters(df_orig, clusters=clusters)
    print(reduced)
    return

    # some features are perfectly correlated and predicting both of them is a
    # waste of resources. We remove correlated features.
    """
    idx = np.triu(np.ones(corrs.shape, dtype=bool), k=1)  # unique correlations
    large = (
        corrs.where(idx)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    )
    large["r"] = large["r"].abs()
    large = large.sort_values(by="r", ascending=False)
    large = large[(large["r"] > 0.9)].sort_values(by=["x", "y"])
    names = list(set(large.x.to_list() + large.y.to_list()))
    corrs = df.loc[:, names].corr()
    """

    infos = []
    # for min_cluster_size in [2, 3, 4]:
    for min_cluster_size in [3]:
        for epsilon in [0.05, 0.1, 0.15, 0.2]:
            clusters = get_cluster_corrs(
                corrs, min_cluster_size=min_cluster_size, epsilon=epsilon
            )
            print("=" * 80)
            print(f"Got {len(clusters)} clusters.")
            sizes = []
            for cluster in clusters:
                sizes.append(len(cluster))
                with pd.option_context("display.max_rows", 100):
                    print("=" * 80)
                    if len(cluster) > 0:
                        # print(cluster.iloc[0, 0])
                        fmt = cluster.to_markdown(
                            tablefmt="simple", index=False, headers=[]
                        )
                        fmt = "\n".join(fmt.split("\n")[1:-1][:20])
                        print(fmt)
            infos.append(
                DataFrame(
                    {
                        "min_size": min_cluster_size,
                        "eps": epsilon,
                        "n_clusters": len(clusters),
                        "cluster_sizes": str(sorted(sizes)),
                    },
                    index=[0],
                )
            )
    info = pd.concat(infos, axis=0, ignore_index=True)
    print(info.to_markdown(tablefmt="simple", index=False))


if __name__ == "__main__":
    explore_corrs()
