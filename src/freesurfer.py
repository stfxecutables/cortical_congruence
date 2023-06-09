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
    DATA,
    load_abide_i_pheno,
    load_abide_ii_pheno,
)

"""
See https://github.com/fphammerle/freesurfer-stats/blob/master/freesurfer_stats/__init__.py
for motivation and reasoning behind code below.
"""

STATFILES = [
    "aseg.stats",
    "wmparc.stats",
    "lh.aparc.a2009s.stats",
    "lh.aparc.stats",
    "rh.aparc.a2009s.stats",
    "rh.aparc.stats",
]
ABIDE_II_EXTRA = [
    "lh.aparc.DKTatlas.stats",  # identical StructNames to lh.aparc.pial.stats
    "lh.aparc.pial.stats",
]


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
        try:
            data = parse_table(lines)
        except ParserError as e:
            text = "\n".join(lines)
            raise RuntimeError(
                f"Could not parse data in {stats}. Offending text:\n{text}"
            ) from e
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

        if "DKTatlas" in str(fname):
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-dkt")
        if "pial" in str(fname):
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-pial")

        df["Struct"] = df["StructName"].str.replace("wm-lh-", "")
        df["Struct"] = df["Struct"].str.replace("wm-rh-", "")
        df["Struct"] = df["Struct"].str.lower()
        df["StructName"] = df["StructName"].str.replace("wm-", "")
        df["StructName"] = df["StructName"].str.replace("Left-", "lh-")
        df["StructName"] = df["StructName"].str.replace("Right-", "rh-")
        df["StructName"] = df["StructName"].str.lower()

        df["hemi"] = df["StructName"].apply(
            lambda s: "left" if "lh-" in s else "right" if "rh-" in s else "both"
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

    try:
        return pd.read_table(StringIO(text), sep="\s+")
    except ParserError:
        return pd.read_table(StringIO(text), sep="\s+", encoding=ABIDE_II_ENCODING)


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


def merge_stats(root: Path, abide2_extra: bool = False) -> DataFrame | None:
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

    # NOTE: None of the structures that lack a hemisphere, that is:
    #
    #      ['3rd-ventricle' '4th-ventricle' 'brain-stem' 'csf' '5th-ventricle'
    #      'wm-hypointensities' 'non-wm-hypointensities' 'optic-chiasm'
    #      'cc_posterior' 'cc_mid_posterior' 'cc_central' 'cc_mid_anterior'
    #      'cc_anterior']
    #
    # have the metrics needed for computing CMC (e.g. SurfArea, ThickAvg, GrayVol)
    df = pd.merge(
        df_hemi.drop(columns=["fname"]),
        df.drop(columns=["fname", "sid", "hemi", "sname", "Struct"]),
        # how="left",
        how="left",
        on="StructName",
    ).sort_values(by=["Volume_mm3", "StructName"])
    df["GrayVol"] = df["GrayVol"].astype(np.float64)
    df["SurfArea"] = df["SurfArea"].astype(np.float64)
    return df


def compute_bilateral_rois(df: DataFrame) -> DataFrame:
    summable_cols = ["Volume_mm3", "GrayVol", "SurfArea", "pseudoVolume"]
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
    df_bilateral["Struct"] = df_bilateral["StructName"].str.replace("bh-", "")
    return df_bilateral


def fill_missing_rois(df: DataFrame, abide: Literal[1, 2], extra: bool) -> DataFrame:
    rois = set(df["StructName"])
    if abide == 1:
        expected_rois = ABIDE_I_ALL_ROIS
    elif abide == 2:
        expected_rois = ABIDE_II_ALL_EXTRA_ROIS if extra else ABIDE_II_ALL_ROIS
    else:
        raise ValueError("Invalid argument to `abide`")

    missing_rois = set(expected_rois).difference(rois)
    if len(missing_rois) == 0:
        return df
    base = df.iloc[0].to_frame().T.copy()
    keeps = ["sid", "sname"]
    meta = base.loc[:, keeps]
    base = base.drop(columns=keeps).applymap(lambda x: np.nan)
    new_rows = []
    for roi in missing_rois:
        new_row = pd.concat([meta, base], axis=1, ignore_index=False)
        new_row.index = [roi]  # type: ignore
        new_row["StructName"] = roi
        new_row["Struct"] = re.sub("[blr]h-", "", roi)
        if "rh" in roi:
            new_row["hemi"] = "right"
        elif "lh" in roi:
            new_row["hemi"] = "left"
        elif "bh" in roi:
            new_row["hemi"] = "both"
        else:
            new_row["hemi"] = "both"
        new_rows.append(new_row)
    return pd.concat([df, *new_rows], axis=0, ignore_index=False)


def compute_bilateral_stats(
    stats: DataFrame, abide: Literal[1, 2], extra: bool
) -> DataFrame:
    """Compute CNC variants and bilateral ROIs"""
    cmc_cols = ["Volume_mm3", "GrayVol", "SurfArea", "pseudoVolume", "ThickAvg"]
    computed_cols = ["CMC_mesh", "CMC_vox", "CMC_asym_mesh", "CMC_asym_vox"]
    all_cols = ["sid", "sname", "StructName", "hemi", "SegId"] + cmc_cols + computed_cols

    unique_names = stats["StructName"].str.replace("[lr]h-", "").unique().tolist()

    # note that stats at this point contains only bilateral ROIs (rh-* or lh-*)
    laterals = []
    for name in unique_names:
        idx = stats["StructName"].apply(lambda s: name in s)
        laterals.append(stats.loc[idx])
    df = pd.concat(laterals, axis=0, ignore_index=True)
    pseudovolume = df["SurfArea"] * df["ThickAvg"]
    df["pseudoVolume"] = pseudovolume

    # compute bilateral ROIs
    df_bilateral = compute_bilateral_rois(df)
    df = pd.concat([df, df_bilateral], axis=0, ignore_index=True)
    df.index = df["StructName"]  # type: ignore

    df = fill_missing_rois(df, abide=abide, extra=extra)
    to_drop = df["SurfArea"].isnull()
    df = df[~to_drop]

    df["CMC_mesh"] = df["GrayVol"] / df["pseudoVolume"]
    df["CMC_vox"] = df["Volume_mm3"] / df["pseudoVolume"]

    df_bi = df[df["hemi"] == "both"].copy()
    df_left = df[df["hemi"] == "left"].copy()
    df_right = df[df["hemi"] == "right"].copy()

    # fill missing ROIs with NaN rows
    # NOTE: PROBLEM IS duplicate names in atlases
    df_left.index = df_left["StructName"].apply(lambda s: s.replace("lh-", ""))  # type: ignore  # noqa
    df_right.index = df_right["StructName"].apply(lambda s: s.replace("rh-", ""))  # type: ignore  # noqa
    cmc_asym_mesh = (
        df_left["CMC_mesh"].sub(df_right["CMC_mesh"], fill_value=np.nan)
    ).abs()
    cmc_asym_vox = (df_left["CMC_vox"].sub(df_right["CMC_vox"], fill_value=np.nan)).abs()
    index_right = df_right["StructName"]
    index_left = df_left["StructName"]
    df_left.index = index_left  # type: ignore
    df_right.index = index_right  # type: ignore

    if (len(df_left) != len(cmc_asym_mesh)) or (len(df_right) != len(df_left)):
        left_rois = set(df_left["Struct"])
        right_rois = set(df_right["Struct"])
        diff1 = left_rois.difference(right_rois)
        diff2 = right_rois.difference(left_rois)
        diff = diff1 if len(diff1) > len(diff2) else diff2
        raise RuntimeError(
            f"Missing ROIs between left vs. right hemispheres: {sorted(diff)}"
        )

    df_left["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_left["CMC_asym_vox"] = cmc_asym_vox.values
    df_right["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_right["CMC_asym_vox"] = cmc_asym_vox.values
    df_bi["CMC_asym_mesh"] = cmc_asym_mesh.values
    df_bi["CMC_asym_vox"] = cmc_asym_vox.values

    df = pd.concat([df_left, df_right, df_bi], axis=0, ignore_index=True).loc[:, all_cols]
    df["SegId"] = df["SegId"].fillna(-1).astype(int)

    return df


def get_subject_meta(sid: str) -> DataFrame:
    """Get: site, dx, dx_dsm_iv, age, sex"""
    abidei = load_abide_i_pheno()
    abide2 = load_abide_ii_pheno()
    df = pd.concat([abidei, abide2], axis=0, ignore_index=True)
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
