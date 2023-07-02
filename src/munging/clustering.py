from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN


class Cluster:
    def __init__(self, data: DataFrame) -> None:
        self.data: DataFrame = data

    @property
    def name(self) -> str:
        names: list[str] = sorted(self.data.iloc[0, :-1].to_list())
        if len(names[0]) <= len(names[1]):
            return names[0]
        return names[1]

    @property
    def names(self) -> list[str]:
        return sorted(set(self.data.x.to_list() + self.data.y.to_list()))

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
