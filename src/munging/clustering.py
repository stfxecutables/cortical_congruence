from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from dataclasses import dataclass
from os.path import commonprefix
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import FactorAnalysis as FA
from tqdm import tqdm

from src.constants import MEMORY


class Cluster:
    def __init__(self, data: DataFrame) -> None:
        self.data: DataFrame = data
        self._name: str | None = None

    @property
    def name(self) -> str:
        # we can cheat and solve this well because we mostly have common suffixes
        if self._name is not None:
            return self._name
        names = np.unique(self.data.iloc[:, :-1].to_numpy()).tolist()
        names_r = ["".join(reversed(name)) for name in names]
        common = "".join(reversed(commonprefix(names_r)))
        count = sum(1 if common in name else 0 for name in names)
        if count / len(names) >= 0.9:
            final = f"CLUST__{common}__{count}/{len(names)}"
            # remove redundant "hemisphere" label in CMC and FS labels
            if "CLUST__h-" in final:
                self._name = final.replace("CLUST__h-", "CLUST__")
                return self._name

        names: list[str] = sorted(self.data.iloc[0, :-1].to_list())
        if len(names[0]) <= len(names[1]):
            self._name = f"CLUST__{names[0]}__{len(names)}"
            return self._name
        self._name = f"CLUST__{names[1]}__{len(names)}"
        return self._name

    @property
    def names(self) -> list[str]:
        return sorted(set(self.data.x.to_list() + self.data.y.to_list()))

    def rename(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.name}: {self.names[:3]} ... )\n"
            f"{self.data.to_markdown(tablefmt='simple', floatfmt='0.3f', index=False)}"
        )

    __repr__ = __str__


def get_cluster_corrs(
    corrs: DataFrame, min_cluster_size: int = 3, epsilon: float = 0.2
) -> list[Cluster]:
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
        cluster.index.name = cluster.columns.name = None
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
    return [Cluster(clust) for clust in clusters if len(clust) > 0]


@MEMORY.cache
def reduce_CMC_clusters(data: DataFrame, clusters: list[Cluster]) -> DataFrame:
    reductions = []
    for cluster in tqdm(clusters, desc="Factor reducing..."):
        name = cluster.name
        df = cluster.data
        feat_names = sorted(set(df.x.to_list() + df.y.to_list()))
        feats = data[feat_names]
        fa = FA(n_components=1, rotation="varimax")
        x = fa.fit_transform(feats.fillna(feats.median()))
        reductions.append(DataFrame(data=x, index=data.index, columns=[name]))
    df = pd.concat(reductions, axis=1)
    return df
