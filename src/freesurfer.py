from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import re
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.errors import ParserError

from src.constants import (
    ABIDE_I_ALL_ROIS,
    ABIDE_II_ALL_EXTRA_ROIS,
    ABIDE_II_ALL_ROIS,
    ABIDE_II_ENCODING,
    ABIDE_II_EXTRA_REMOVE,
    ADHD200,
    ADHD200_ALL_ROIS,
    BASE_STATFILES,
    DATA,
    load_abide_i_pheno,
    load_abide_ii_pheno,
    load_adhd200_pheno,
)
from src.enumerables import FreesurferStatsDataset
from src.munging.fs_stats.parse import FreesurferStats, MetaData

"""
See https://github.com/fphammerle/freesurfer-stats/blob/master/freesurfer_stats/__init__.py
for motivation and reasoning behind code below.
"""

ABIDE_II_EXTRA = [
    "lh.aparc.DKTatlas.stats",  # identical StructNames to lh.aparc.pial.stats
    "lh.aparc.pial.stats",
]


@dataclass
class Subject:
    root: Path
    id: str
    statfiles: list[Path]

    @staticmethod
    def from_root(root: Path) -> Subject:
        stats = root.rglob("*.stats")


def is_standard(file: Path) -> bool:
    return not (
        file.name
        in [
            "lh.aparc.DKTatlas40.stats",
            "lh.curv.stats" "lh.w-g.pct.stats" "rh.aparc.DKTatlas40.stats",
            "rh.curv.stats" "rh.w-g.pct.stats",
        ]
    )


def collect_struct_names(root: Path, abide2_extra: bool = False) -> set:
    statfiles = STATFILES
    if abide2_extra:
        statfiles = STATFILES + ABIDE_II_EXTRA

    stats = [root / stat for stat in statfiles]
    available = list(map(lambda p: p.exists(), stats))
    if not all(available):
        return None

    dfs, df_hemi = [], []
    for stat in stats:
        fs = FreesurferStats.from_statsfile(stat)
        df = fs.to_subject_table()
        if fs.has_area:
            df_hemi.append(df)
        else:
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True).drop(columns="parent")
    df_hemi = pd.concat(df_hemi, axis=0, ignore_index=True).drop(columns="parent")
    names = set(df["StructName"]).union(df_hemi["StructName"])
    # remove useless structures with no info to compute CMC
    names = set([n for n in names if re.match("[lr]h-", n) is not None])
    names = set([n.replace("lh-", "").replace("rh-", "") for n in names])
    return names


def get_subject_meta(sid: str) -> DataFrame:
    """Get: site, dx, dx_dsm_iv, age, sex"""
    abidei = load_abide_i_pheno()
    abide2 = load_abide_ii_pheno()
    adhd = load_adhd200_pheno()
    df = pd.concat([abidei, abide2, adhd], axis=0, ignore_index=True)
    meta = df[df["sid"].isin([int(sid)])]
    if isinstance(meta, Series):
        meta = meta.to_frame()
    if len(meta) == 0:
        raise ValueError(f"SID {sid} not found in ABIDE-I or ABIDE-II phenotypic data")
    if len(meta) > 1:
        raise ValueError(f"Duplicate SIDs in ABIDE-I and ABIDE-II for sid: {sid}\n{meta}")
    return meta.drop(columns="sid")


def bilateral_stats_to_row(stats: DataFrame) -> DataFrame:
    df = stats.drop(
        columns=[
            "sid",
            "sname",
            "hemi",
            "SegId",
            "Volume_mm3",
            "GrayVol",
            "SurfArea",
            "pseudoVolume",
            "ThickAvg",
        ]
    )
    meta = stats.loc[0, ["sid", "sname"]].to_frame().T  # type: ignore
    df.index = df["StructName"]  # type: ignore
    df.drop(columns="StructName", inplace=True)

    row = df.unstack()
    row.index = map(lambda items: f"{items[1]}__{items[0]}", row.index.to_list())  # type: ignore  # noqa
    row = row.to_frame().T  # type: ignore
    df = pd.concat([meta, row], axis=1, ignore_index=False)
    meta = get_subject_meta(int(df.loc[0, "sid"]))  # type: ignore
    for col in meta.columns.to_list():
        df.insert(1, col, meta[col].values)

    return df


if __name__ == "__main__":
    PARENT = ROOT / "data/ABIDE-I/Caltech_0051457/stats"
    TEST = ROOT / "data/ABIDE-I/Caltech_0051457/stats/lh.aparc.stats"

    collect_struct_names(ADHD200)
    df = merge_stats(PARENT)
    pd.options.display.max_rows = 300
    print(df.drop(columns=["Struct"]))
    print(df)
    sys.exit()

    df = compute_bilateral_stats(df)
    print(df)
    row = bilateral_stats_to_row(df)
    print(row)
    sys.exit()
    stats = sorted(PARENT.glob("*.stats"))
    dfs, df_hemi = [], []
    for stat in stats:
        fs = FreesurferStats.from_statsfile(stat)
        df = fs.to_subject_table()
        if fs.has_area:
            df_hemi.append(df)
        else:
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True).drop(columns="parent")
    df_hemi = pd.concat(df_hemi, axis=0, ignore_index=True).drop(columns="parent")
    df_all = pd.merge(
        df.drop(columns=["fname"]),
        df_hemi.drop(columns=["fname", "sid", "hemi", "sname", "Struct"]),
        how="left",
        on="StructName",
    )
    pd.options.display.max_rows = 300
    print(df_all.drop(columns=["Struct", "sname", "sid"]))
    sys.exit()

    uq_fnames = df_hemi["fname"].unique().tolist()
    print(uq_fnames)
    # reg = r"([lr]h)."
    # df_hemi["hemi"] = df_hemi["fname"].apply(lambda s: f"wm-{re.search(reg, s)[1]}-")
    # df_hemi["StructName"] = df_hemi["hemi"] + df_hemi["StructName"]
    print(df_hemi)

    # print(f"Unique fles: {df['fname'].unique()}")

    # uq_fnames = df_hemi["fname"].unique().tolist()
    # examples = []
    # for unq in uq_fnames:
    #     examples.append(df_hemi.loc[df_hemi.fname.isin([unq])].iloc[:2, :])
    # df_hemi_sample = pd.concat(examples, axis=0, ignore_index=True)
    # # print(df_area.drop(columns="parent"))
    # print(df_hemi_sample)
    # print(f"Unique fles: {uq_fnames}")

    names1 = set(df["Struct"].to_list())
    names2 = set(df_hemi["Struct"].to_list())
    print("Shared StructNames:")
    print(set(names1).intersection(names2))
    sys.exit()
    print("aseg.stats + wmparc.stats structures")
    for name in sorted(names1):
        print(name)

    print("=" * 80)
    print("Hemispheric structures")
    for name in sorted(names2):
        print(name)

    # structs = ["wm-lh-caudalanteriorcingulate", "wm-rh-caudalanteriorcingulate", "caudalanteriorcingulate"]
    # print(df.loc[df.StructName.isin(structs)])
    # print(df_area.loc[df_area.StructName.isin(structs)])
