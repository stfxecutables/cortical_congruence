from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import (
    ABIDE_II_ENCODING,
    ALL_STATSFILES,
    CACHED_RESULTS,
    DATA,
    HCP_FEATURE_INFO,
    MEMORY,
    load_abide_i_pheno,
    load_abide_ii_pheno,
    load_adhd200_pheno,
)
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus
from src.munging.clustering import get_cluster_corrs, reduce_CMC_clusters
from src.munging.fs_stats.tabularize import (
    add_bilateral_stats,
    compute_cmcs,
    to_wide_subject_table,
)
from src.munging.hcp import cleanup_HCP_phenotypic, reduce_HCP_clusters


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
        """Makes the stats file into a consistent table with rows for each ROI
        in the stats file, and columns:

            sid, sname, parent, fname, Struct, StructName, hemi

        plus whatever other columns exist in the actual file.

        The `Struct` column is the unadorned structure name, e.g. with "wm-"
        and "lh-" indicators removed. The "StructName" column removes any
        useless indicators (e.g wm-) and changes inconsistent indicators (e.g.
        "Left-") to be the same (e.g. "lh-).
        """
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

        df["hemi"] = df["fname"].apply(
            lambda name: "left"
            if name.startswith("lh.")
            else "right"
            if name.startswith("rh.")
            else "unknown"
        )

        sfname = str(fname)
        if "DKTatlas" in sfname:
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-dkt")
        if "a2009s" in sfname:
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-a2009kt")
        if "pial" in sfname:
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-pial")

        # NOTE: this is to prevent duplicate structnames! in e.g. wmparc.stats
        # and *h.aparc.stats files, or similar which can happen sometimes for
        # some reason
        if sfname == "wmparc.stats":
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-wmparc")
        elif sfname == "aseg.stats":
            df["StructName"] = df["StructName"].apply(lambda s: f"{s}-aseg")

        df["Struct"] = df["StructName"].apply(lambda s: re.sub("[lrb]h-", "", s))
        df["Struct"] = df["Struct"].str.lower()
        df["Struct"] = df["Struct"].str.replace("left-", "")
        df["Struct"] = df["Struct"].str.replace("right-", "")

        # Handle bad StructNames. Bad struct names come mostly from wmparc.stats,
        # which labels some structs as wm-lh- or wm-rh- , or Left- or Right- all
        # very inconsistently. This is annoying ans we want to remove these bad
        # naming conventions so rh-, lh-, or bh- are always at the front of the
        # name.
        #
        # Also, there is one very stupid set of names that makes no sense:
        #
        #     WM-hypointensities
        #     Left-WM-hypointensities
        #     Right-WM-hypointensities
        #     non-WM-hypointensities
        #     Left-non-WM-hypointensities
        #     Right-non-WM-hypointensities
        #
        # The relation of left to right here is unclear. The problem is these
        # rename to:
        #
        #     bh-wm-hypointensities
        #     lh-wm-hypointensities
        #     rh-wm-hypointensities
        #     bh-non-wm-hypointensities
        #     lh-non-wm-hypointensities
        #     rh-non-wm-hypointensities
        #
        # and then the names
        #
        #     lh-wm-hypointensities
        #     rh-wm-hypointensities
        #
        # combine into `bh-wm-hypointensities`, which is of course a dupe. The
        # same happens for bh-non-wm-hypointensities. We resolve this by naming
        #
        #     WM-hypointensities      -->   all-wm-hypointensities
        #     non-WM-hypointensities  -->   all-non-wm-hypointensities

        df["StructName"] = df["StructName"].str.lower()
        df["StructName"] = df["StructName"].apply(
            lambda s: "all-wm-hypointensities-aseg"
            if s == "wm-hypointensities-aseg"
            else s
        )
        df["StructName"] = df["StructName"].apply(
            lambda s: "all-non-wm-hypointensities-aseg"
            if s == "non-wm-hypointensities-aseg"
            else s
        )

        idx = df["hemi"] == "unknown"
        unknown_hemis = (
            df["StructName"]
            .loc[idx]
            .apply(
                lambda name: "left"
                if ("left-" in name or "lh-" in name)
                else "right"
                if ("right-" in name or "rh-" in name)
                else "both"
            )
        )
        df.loc[idx, "hemi"] = unknown_hemis
        df["StructName"] = df["StructName"].str.replace("left-", "")
        df["StructName"] = df["StructName"].str.replace("right-", "")
        df["StructName"] = df["StructName"].str.replace("lh-", "")
        df["StructName"] = df["StructName"].str.replace("rh-", "")
        df["StructName"] = (
            df["hemi"].apply(lambda s: {"left": "lh-", "right": "rh-", "both": "bh-"}[s])
            + df["StructName"]
        )

        cols.remove("StructName")
        df = df.loc[
            :, ["sid", "sname", "parent", "fname", "StructName", "Struct", "hemi"] + cols
        ]

        # Handle trash HBN data not having unique SIDs
        if "HBN" in str(self.meta.filename.parent):
            df["sid"] = df["sname"]
        # Handle trash ABIDE-I data having wrong SID column values
        if "ABIDE-I" in str(self.meta.filename.parent):
            if "ABIDE-II" not in str(self.meta.filename.parent):
                corrected = int(re.search(r".*_(\d+)", self.meta.subjectname)[1])  # type: ignore  # noqa
                df["sid"] = str(corrected)
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
        return pd.read_table(StringIO(text), sep="\s+")  # noqa # type: ignore
    except ParserError:
        return pd.read_table(
            StringIO(text), sep="\s+", encoding=ABIDE_II_ENCODING  # noqa # type: ignore
        )


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
        infos.append(parse_table_metadata_lines(triple))  # type: ignore
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


def to_frame(stat: Path) -> DataFrame:
    return FreesurferStats.from_statsfile(stat).to_subject_table()


def reformat_HPC(df: DataFrame) -> DataFrame:
    path = cast(Path, FreesurferStatsDataset.HCP.freesurfer_files())
    df = pd.read_csv(path)
    df.rename(
        columns={"Subject": "sid", "Gender": "sex", "Release": "release"}, inplace=True
    )
    df.rename(columns=str.lower, inplace=True)
    df.rename(columns=lambda s: s.replace("fs_r_", "rh-"), inplace=True)
    df.rename(columns=lambda s: s.replace("fs_l_", "lh-"), inplace=True)
    drop_reg = "curv|foldind|thckstd|numvert|_range|_min|_max|_std|intens_mean|_vox"
    keep_reg = "sid|gender|_grayvol|_area|_thck"
    df = df.filter(regex=keep_reg).copy()
    df["sid"] = df["sid"].astype(str)
    df["sname"] = df["sid"].astype(str)
    df["parent"] = str(path.parent)
    df["fname"] = path.name
    meta_cols = ["sid", "sname", "parent", "fname"]
    cols = list(set(df.columns).difference(meta_cols))
    df = df.loc[:, meta_cols + cols].copy()
    df = df.melt(id_vars=meta_cols, var_name="metric").sort_values(by=["sid", "metric"])

    vols = (
        df[df["metric"].str.contains("grayvol")]
        .sort_values(by=["sid", "metric"])
        .reset_index(drop=True)
    )
    areas = (
        df[df["metric"].str.contains("area")]
        .copy()
        .sort_values(by=["sid", "metric"])
        .reset_index(drop=True)
    )
    thick = (
        df[df["metric"].str.contains("thck")]
        .copy()
        .sort_values(by=["sid", "metric"])
        .reset_index(drop=True)
    )
    vols.loc[:, "StructName"] = vols["metric"].str.replace("_grayvol", "")
    areas.loc[:, "StructName"] = areas["metric"].str.replace("_area", "")
    thick.loc[:, "StructName"] = thick["metric"].str.replace("_thck", "")

    vols.drop(columns="metric", inplace=True)
    areas.drop(columns="metric", inplace=True)
    thick.drop(columns="metric", inplace=True)

    vols.rename(columns={"value": "GrayVol"}, inplace=True)
    areas.rename(columns={"value": "SurfArea"}, inplace=True)
    thick.rename(columns={"value": "ThickAvg"}, inplace=True)
    vols["SurfArea"] = areas["SurfArea"]
    vols["ThickAvg"] = thick["ThickAvg"]

    df = vols.copy()
    df["Struct"] = df["StructName"].str.replace("lh-", "")
    df["Struct"] = df["Struct"].str.replace("rh-", "")
    df["hemi"] = df["StructName"].apply(
        lambda s: "left" if s.startswith("lh") else "right"
    )
    df = df.loc[
        :, meta_cols + ["StructName", "Struct", "hemi", "GrayVol", "SurfArea", "ThickAvg"]
    ].copy()
    return df


def load_HCP_CMC_table() -> DataFrame:
    path: Path = cast(Path, FreesurferStatsDataset.HCP.freesurfer_files())
    df = pd.read_csv(path)
    df = reformat_HPC(df)
    df["pseudoVolume"] = df["ThickAvg"] * df["SurfArea"]
    df = add_bilateral_stats(df)
    df = compute_cmcs(df)
    return df


def load_phenotypic_data(
    dataset: FreesurferStatsDataset, pheno: PhenotypicFocus = PhenotypicFocus.Reduced
) -> DataFrame:
    """Get: site, dx, dx_dsm_iv, age, sex"""

    source = dataset.phenotypic_file()
    sep = "," if source.suffix == ".csv" else "\t"
    df = pd.read_csv(source, sep=sep)
    if dataset is FreesurferStatsDataset.HCP:
        extra = pheno.hcp_dict_file()
        feats = pd.read_csv(extra)
        return cleanup_HCP_phenotypic(df, extra=feats)

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


@MEMORY.cache
def load_HCP_complete(
    *,
    focus: PhenotypicFocus = PhenotypicFocus.Reduced,
    reduce_targets: bool,
    reduce_cmc: bool,
) -> DataFrame:
    df = load_HCP_CMC_table()
    df = to_wide_subject_table(df)
    pheno = load_phenotypic_data(FreesurferStatsDataset.HCP, focus)
    shared = list(
        set(df.filter(regex="FS").columns).intersection(pheno.filter(regex="FS").columns)
    )
    df = pd.concat([df, pheno.drop(columns=shared)], axis=1)
    df.rename(columns=lambda col: col.replace("FS__PVOL", "CMC__pvol"), inplace=True)
    if reduce_targets:
        if focus is not PhenotypicFocus.All:
            raise NotImplementedError("Target reduction implemented only for full data")
        targs = df.filter(regex="TARGET__")
        cols = targs.columns
        targs = targs.rename(columns=lambda s: s.replace("TARGET__", ""))
        corrs = targs.corr()  # most correlations differ at most by 0.1 with spearman
        clusters = get_cluster_corrs(corrs)
        targs_reduced = reduce_HCP_clusters(data=targs, clusters=clusters)
        targs_reduced.rename(columns=lambda s: f"TARGET__{s}", inplace=True)
        others = df.drop(columns=cols, errors="ignore")
        df = pd.concat([others, targs_reduced], axis=1)
    if reduce_cmc:
        cmcs = df.filter(regex="CMC__")
        cols = cmcs.columns
        cmcs = cmcs.rename(columns=lambda s: s.replace("CMC__", ""))
        corrs = cmcs.corr()
        clusters = get_cluster_corrs(corrs, min_cluster_size=3, epsilon=0.2)
        cmc_reduced = reduce_CMC_clusters(data=cmcs, clusters=clusters)
        others = df.drop(columns=cols)
        cmc_reduced.rename(columns=lambda s: f"CMC__{s}", inplace=True)
        df = pd.concat([others, cmc_reduced], axis=1)

    return df


if __name__ == "__main__":
    # pd.options.display.max_rows = 500
    # load_HCP_complete.clear()
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=False, reduce_cmc=True
    )
    dfr = df.filter(regex="CMC|TARGET|DEMO")
    corrs = (
        dfr.corr()
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    )
    corrs["abs"] = corrs["r"].abs()
    corrs = corrs.sort_values(by="abs", ascending=False).drop(columns="abs")
    corrs = corrs[corrs["r"] < 1.0]
    cmc_corrs = corrs[
        (corrs.x.str.contains("CMC") & corrs.y.str.contains("TARGET"))
        | (corrs.x.str.contains("TARGET") & corrs.y.str.contains("CMC"))
    ]
    print(df)

    # for dataset in [
    #     FreesurferStatsDataset.ABIDE_I,
    #     FreesurferStatsDataset.ABIDE_II,
    #     FreesurferStatsDataset.ADHD_200,
    #     FreesurferStatsDataset.HBN,
    # ]:
    #     df = compute_CMC_table(dataset)
    #     df = to_wide_subject_table(df)
    #     print(df)

    # df = load_HCP_CMC_table()
    # df = to_wide_subject_table(df)

    # print(df)
