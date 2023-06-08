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
    Optional,
    Sequence,
    TextIO,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series

from src.constants import DATA

"""
See https://github.com/fphammerle/freesurfer-stats/blob/master/freesurfer_stats/__init__.py
for motivation and reasoning behind code below.
"""


@dataclass
class ColumnInfo:
    shortname: str  # ColHeader
    name: str  # FileName
    unit: str  # Units
    index: int


@dataclass
class MetaData:
    subjectname: str
    filename: Path
    annot: Path


@dataclass
class Subject:
    root: Path
    id: str
    statfiles: list[Path]

    @staticmethod
    def from_root(root: Path) -> Subject:
        stats = root.rglob("*.stats")


@dataclass
class FreesurferStats:
    root: Path
    sid: str
    columns: list[ColumnInfo]
    meta: MetaData
    data: DataFrame

    @staticmethod
    def from_statsfile(stats: Path) -> FreesurferStats:
        root = stats.parent
        try:
            with open(stats, "r") as handle:
                lines = handle.readlines()
        except Exception as e:
            raise RuntimeError(f"Could not read lines from file: {stats}") from e

        meta = parse_metadata(lines, stats)
        data = parse_table(lines)
        columns = parse_table_metadata(lines)
        result = re.search(r"\d+", meta.subjectname)
        if result is None:
            sid = meta.subjectname
        else:
            sid = result[0]

        return FreesurferStats(
            root=root,
            sid=sid,
            columns=columns,
            meta=meta,
            data=data,
        )

    def to_subject_table(self) -> DataFrame:
        # below make table with columns
        df = self.data.copy()
        if "Index" in df.columns:
            df.drop(columns="Index", inplace=True)
        cols = list(df.columns)
        fname = self.meta.filename.name

        df["sid"] = self.sid
        df["sname"] = self.meta.subjectname
        df["parent"] = self.meta.filename.parent
        df["fname"] = fname

        if fname.startswith("lh.") or fname.startswith("rh."):
            reg = r"([lr]h)."
            hemi = df["fname"].apply(lambda s: f"wm-{re.search(reg, s)[1]}-")  # type: ignore # noqa
            df["StructName"] = hemi + df["StructName"]
        else:
            pass

        df["Struct"] = df["StructName"].str.replace("wm-lh-", "")
        df["Struct"] = df["Struct"].str.replace("wm-rh-", "")
        df["Struct"] = df["Struct"].str.lower()
        df["StructName"] = df["StructName"].str.replace("wm-", "")
        df["StructName"] = df["StructName"].str.replace("Left-", "lh-")
        df["StructName"] = df["StructName"].str.replace("Right-", "rh-")
        df["StructName"] = df["StructName"].str.lower()
        df["hemi"] = df["StructName"].apply(
            lambda s: "left" if "lh-" in s else "right" if "rh-" in s else "NA"
        )

        cols.remove("StructName")
        df = df.loc[
            :, ["sid", "sname", "parent", "fname", "StructName", "Struct", "hemi"] + cols
        ]
        return df

    @property
    def has_area(self) -> bool:
        return "SurfArea" in self.data.columns

    def __str__(self) -> str:
        fmt = []
        fmt.append(f"{self.sid} @ {self.root.resolve()}")
        fmt.append(self.data.head().to_markdown(tablefmt="simple"))
        return "\n".join(fmt)

    __repr__ = __str__


def parse_comment(line: str) -> str:
    if not line.startswith("# "):
        raise ValueError("Not a comment line")
    return line[2:]


def parse_colnames(line: str) -> list[str]:
    if not line.startswith("# "):
        raise ValueError("Not a table header line")
    line = line[2:].rstrip()
    return line.split(" ")


def parse_metadata(lines: list[str], fname: Path) -> MetaData:
    end = -1
    for i, line in enumerate(lines):
        if line.startswith("# NTableCols"):
            end = i
            break
    lines = lines[:end]

    subject, annot = "", ""
    for line in lines:
        if "subjectname" in line:
            subject = line.replace("# subjectname ", "").strip()
        elif "# AnnotationFile " in line:
            annot = line.replace("# AnnotationFile ", "").strip()
        else:
            continue
    return MetaData(
        subjectname=subject,
        filename=Path(fname).relative_to(DATA),
        annot=Path(annot),
    )


def parse_table(lines: list[str]) -> DataFrame:
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("# ColHeaders"):
            start = i
    lines = lines[start:]
    lines[0] = lines[0].replace("# ColHeaders ", "")
    text = "\n".join(lines)

    return pd.read_table(StringIO(text), sep="\s+")


def parse_table_metadata(lines: list[str]) -> list[ColumnInfo]:
    lines = list(filter(lambda line: "# TableCol" in line, lines))
    indices = list(range(0, len(lines), 3))
    triples = []
    for i in range(len(indices) - 1):
        start = indices[i]
        stop = indices[i + 1]
        triples.append(tuple(lines[start:stop]))
    infos = []
    for triple in triples:
        infos.append(parse_table_metadata_lines(triple))
    return infos


def parse_table_metadata_lines(triple: tuple[str, str, str]) -> ColumnInfo:
    shortname, name, unit, index = "", "", "", ""
    for line in triple:
        if not line.startswith("# TableCol"):
            raise ValueError("Not a table column header line")
        line = line.replace("# TableCol  ", "")
        try:
            index, fieldname, *values = line.split(" ")
            value = " ".join(values).strip()
        except ValueError as e:
            raise RuntimeError(f"Could not parse line: {line}") from e

        if fieldname == "ColHeader":
            shortname = value
        elif fieldname == "FieldName":
            name = value
        elif fieldname == "Units":
            unit = value
        else:
            raise ValueError(f"Unrecognized table header line: `{line}`")
    return ColumnInfo(shortname=shortname, name=name, unit=unit, index=int(index))


PARENT = ROOT / "data/ABIDE-I/Caltech_0051457/stats"
TEST = ROOT / "data/ABIDE-I/Caltech_0051457/stats/lh.aparc.stats"


def merge_stats(root: Path) -> DataFrame:
    stats = sorted(root.glob("*.stats"))
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
    df = pd.merge(
        df.drop(columns=["fname"]),
        df_hemi.drop(columns=["fname", "sid", "hemi", "sname", "Struct"]),
        # how="left",
        how="inner",
        on="StructName",
    )
    df["GrayVol"] = df["GrayVol"].astype(np.float64)
    df["SurfArea"] = df["SurfArea"].astype(np.float64)
    return df


def compute_bilateral_stats(stats: DataFrame) -> DataFrame:
    """Compute CNC variants and bilateral ROIs"""
    summable_cols = ["Volume_mm3", "GrayVol", "SurfArea", "pseudoVolume"]
    cmc_cols = summable_cols + ["ThickAvg"]
    computed_cols = ["CMC_mesh", "CMC_vox", "CMC_asym_mesh", "CMC_asym_vox"]
    keep_cols = ["sid", "sname", "StructName", "hemi", "SegId"] + cmc_cols
    all_cols = ["sid", "sname", "StructName", "hemi", "SegId"] + cmc_cols + computed_cols

    unique_names = stats["StructName"].str.replace("[lr]h-", "").unique().tolist()

    # note that stats at this point contains only bilateral ROIs
    laterals = []
    for name in unique_names:
        idx = stats["StructName"].apply(lambda s: name in s)
        laterals.append(stats.loc[idx])
    df = pd.concat(laterals, axis=0, ignore_index=True)
    pseudovolume = df["SurfArea"] * df["ThickAvg"]
    df["pseudoVolume"] = pseudovolume

    # compute bilateral ROIs
    df_bilateral = (
        df.groupby("Struct")
        .sum()
        .loc[:, summable_cols]
        .reset_index()
        .rename(columns={"Struct": "StructName"})
    )
    df_bilateral["StructName"] = df_bilateral["StructName"].apply(lambda s: f"bh-{s}")
    df_bilateral["sid"] = df["sid"].iloc[0]
    df_bilateral["sname"] = df["sname"].iloc[0]
    df_bilateral["hemi"] = "both"
    df = pd.concat([df, df_bilateral], axis=0, ignore_index=True)
    df.index = df["StructName"]  # type: ignore

    df["CMC_mesh"] = df["GrayVol"] / df["pseudoVolume"]
    df["CMC_vox"] = df["Volume_mm3"] / df["pseudoVolume"]

    df_bi = df[df["hemi"] == "both"].copy()
    df_left = df[df["hemi"] == "left"].copy()
    df_left.index = df_left["StructName"].apply(lambda s: s.replace("lh-", ""))
    df_right = df[df["hemi"] == "right"].copy()
    df_right.index = df_right["StructName"].apply(lambda s: s.replace("rh-", ""))
    cmc_asym_mesh = (df_left["CMC_mesh"] - df_right["CMC_mesh"]).abs()
    cmc_asym_vox = (df_left["CMC_vox"] - df_right["CMC_vox"]).abs()
    df_left.index = df_left["StructName"].apply(lambda s: f"lh-{s}")
    df_right.index = df_left["StructName"].apply(lambda s: f"rh-{s}")
    df_left["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_left["CMC_asym_vox"] = cmc_asym_vox.values
    df_right["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_right["CMC_asym_vox"] = cmc_asym_vox.values
    df_bi["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_bi["CMC_asym_vox"] = cmc_asym_vox.values

    df = pd.concat([df_left, df_right, df_bi], axis=0, ignore_index=True).loc[:, all_cols]
    df["SegId"] = df["SegId"].fillna(-1).astype(int)

    return df


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
    meta = stats.loc[0, ["sid", "sname"]].to_frame().T
    df.index = df["StructName"]
    df.drop(columns="StructName", inplace=True)

    row = df.unstack()
    row.index = map(lambda items: f"{items[1]}__{items[0]}", row.index.to_list())
    row = row.to_frame().T
    df = pd.concat([meta, row], axis=1, ignore_index=False)
    return df


if __name__ == "__main__":
    df = merge_stats(PARENT)
    pd.options.display.max_rows = 300
    print(df.drop(columns=["Struct"]))

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
