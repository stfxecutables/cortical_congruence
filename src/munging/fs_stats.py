from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Callable
from pandas.errors import ParserError
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from src.constants import ABIDE_II_ENCODING, ALL_STATSFILES, CACHED_RESULTS, DATA
from src.enumerables import FreesurferStatsDataset


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
                corrected = int(re.search(r".*_(\d+)", self.meta.subjectname)[1])
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


def tabularize_all_stats_files(dataset: FreesurferStatsDataset) -> DataFrame:
    if dataset not in [
        FreesurferStatsDataset.ABIDE_I,
        FreesurferStatsDataset.ABIDE_II,
        FreesurferStatsDataset.ADHD_200,
        FreesurferStatsDataset.HBN,
    ]:
        raise FileNotFoundError(
            f"Dataset {dataset} does not have FreeSurfer data in *.stats files."
        )

    out = CACHED_RESULTS / f"{dataset.value}__stats.parquet"
    if out.exists():
        return pd.read_parquet(out)
    root = dataset.root()
    stats = sorted(list(root.rglob("*.stats")))
    stats = [p for p in stats if p.name in ALL_STATSFILES]
    dfs = process_map(
        to_frame, stats, desc=f"Parsing *.stats files from {root}", chunksize=1
    )

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df["GrayVol"] = df["GrayVol"].astype(np.float64)
    df["SurfArea"] = df["SurfArea"].astype(np.float64)
    df["pseudoVolume"] = df["ThickAvg"] * df["SurfArea"]
    df["parent"] = df["parent"].apply(str)

    df.to_parquet(out)
    print(f"Cached stats DataFrame at {out}")
    return df


def add_bilateral_stats(stats_df: DataFrame) -> DataFrame:
    vols = ["Volume_mm3", "GrayVol", "pseudoVolume"]
    other = stats_df.columns.to_list()
    for vol in vols:
        other.remove(vol)
    meta_cols = ["sid", "sname", "parent", "fname"]

    df = stats_df.copy()

    left = (
        df[df.hemi == "left"]
        .sort_values(by=["sid", "StructName"])
        .reset_index(drop=True)  # nofmt
    )
    right = (
        df[df.hemi == "right"]
        .sort_values(by=["sid", "StructName"])
        .reset_index(drop=True)
    )
    if len(left) != len(right):
        raise ValueError("Unequal hemispheres. Fill missing did not work.")
    if not all(left["StructName"].str.replace("lh-", "rh-") == right["StructName"]):
        raise ValueError("Asymmetry in left and right tables")

    vol_sums = left[vols] + right[vols]
    df_bi = pd.concat([left[other], vol_sums], axis=1)
    df_bi["hemi"] = "both"

    df_bi["StructName"] = df_bi["StructName"].apply(lambda s: re.sub("[rl]h-", "bh-", s))

    df_filled = pd.concat([df, df_bi], axis=0)

    return df_filled


def detect_table_issues(df: DataFrame) -> tuple[bool, DataFrame | None, list[str]]:
    dfs = [d for _, d in df.groupby("sid")]
    for df in dfs:
        dupes = df["StructName"][df["StructName"].duplicated()]
        names = df["StructName"].unique()
        bad_names = []
        for name in names:
            if "bh-" in name:
                if "lh-" in name or "rh-" in name:
                    bad_names.append(name)
        if len(dupes) > 0:
            print("Dupes:")
            print(sorted(dupes))
        if len(bad_names) > 0:
            print("Bad Names:")
            print(bad_names)
        if len(dupes) > 0 or len(bad_names) > 0:
            return (
                True,
                df[df["StructName"].isin(dupes)].sort_values(by="StructName"),
                bad_names,
            )

    return False, None, []


def detect_table_symmetry(df: DataFrame) -> None:
    dfs = [d for _, d in df.groupby("sid")]
    for df in dfs:
        left = (
            df[df["hemi"] == "left"].sort_values(by="StructName").reset_index(drop=True)
        )
        right = (
            df[df["hemi"] == "right"].sort_values(by="StructName").reset_index(drop=True)
        )
        left["id"] = left.sid + "__" + left.StructName
        right["id"] = right.sid + "__" + right.StructName

        if len(left) != len(right):
            pd.options.display.max_rows = 500
            print("Fucked table")
            print(df.iloc[:, :10].sort_values(by="StructName"))
            print("Something wrong:")
            print(set(left["id"]).symmetric_difference(right["id"]))
            raise ValueError("Asymmetric table")

        idx = left["StructName"].str.replace("lh-", "rh-") != right["StructName"]
        left_problems = left[idx]
        right_problems = right[idx]
        if len(left_problems) > 0 or len(right_problems) > 0:
            pd.options.display.max_rows = 500
            print("Asymmetry in StructNames found:")
            print("Left:")
            print(left_problems)
            print("Right:")
            print(right_problems)
            raise ValueError("StructName asymmetries")


def _add_missing_rows(names: list[str]) -> Callable[[DataFrame], DataFrame]:
    def closure(df: DataFrame) -> DataFrame:
        existing = df["StructName"].unique().tolist()
        missing = sorted(set(names).difference(existing))

        if len(df) > len(names):
            is_corrupt, dupes, bad_names = detect_table_issues(df)
            print("Duplicates:\n")
            print(dupes)
            raise ValueError("Complete and utter BS due to dupes above.")

        if len(missing) == 0:
            assert len(df) == len(names), "Utter bullshit"
            return df.copy()
        n_rows = len(missing)
        n_cols = len(df.columns)
        df2 = DataFrame(
            data=np.full([n_rows, n_cols], fill_value=np.nan), columns=df.columns
        )
        df2["StructName"] = missing
        # only replace lh- or rh- names so bh- synthetic rois remain
        df2["Struct"] = df2["StructName"].apply(lambda s: re.sub("[rl]h-", "", s))
        df2["sid"] = df["sid"].iloc[0]
        df2["sname"] = df["sname"].iloc[0]
        df2["hemi"] = df2["StructName"].apply(
            lambda s: "left"
            if s.startswith("lh-")
            else "right"
            if s.startswith("rh-")
            else "both"
        )
        df_filled = pd.concat([df, df2], axis=0, ignore_index=True)
        detect_table_symmetry(df_filled)
        if len(df_filled) != len(names):
            dupe = df_filled["StructName"][df_filled["StructName"].duplicated()]
            raise ValueError(
                f"Duplicates:\n{df_filled[df_filled['StructName'].isin([dupe])]}"
            )

        return df_filled

    return closure


def fill_missing_rois(df_stats: DataFrame) -> DataFrame:
    df = df_stats.copy()
    is_corrupt, dupes, bad_names = detect_table_issues(df)
    if is_corrupt:
        print(dupes)
        raise ValueError("Duplicates.")
    left = df[df["hemi"] == "left"]
    right = df[df["hemi"] == "right"]
    both = df[df["hemi"] == "both"]

    left_unq = left["StructName"].unique().tolist()
    right_unq = right["StructName"].unique().tolist()
    both_unq = both["StructName"].unique().tolist()

    lrs = [l.replace("lh-", "rh-") for l in left_unq]
    rls = [r.replace("rh-", "lh-") for r in right_unq]
    unq_names = sorted(set(both_unq + left_unq + right_unq + lrs + rls))

    # lefts = sorted(filter(lambda name: "lh-" in name, unq_names))
    # rights = sorted(filter(lambda name: "rh-" in name, unq_names))

    df = df.groupby("sid").apply(_add_missing_rows(unq_names))
    df.index = range(len(df))  # type: ignore
    left = df[df["hemi"] == "left"]
    right = df[df["hemi"] == "right"]
    detect_table_symmetry(df)
    return df


def compute_cmcs(bilateral_df: DataFrame) -> DataFrame:
    df = bilateral_df.copy()

    df["CMC_mesh"] = df["GrayVol"] / df["pseudoVolume"]
    df["CMC_vox"] = df["Volume_mm3"] / df["pseudoVolume"]
    df["CMC_vox"].replace(np.inf, np.nan, inplace=True)

    left = df[df["hemi"] == "left"].sort_values(by=["sid", "StructName"])
    right = df[df["hemi"] == "right"].sort_values(by=["sid", "StructName"])
    if len(left) != len(right):
        raise ValueError("Invalid")
    left.index = range(len(left))
    right.index = range(len(right))
    cmc_asyms = left.filter(regex="CMC") - right.filter(regex="CMC")
    return df


if __name__ == "__main__":
    for dataset in [
        FreesurferStatsDataset.ABIDE_I,
        FreesurferStatsDataset.ABIDE_II,
        FreesurferStatsDataset.ADHD_200,
        FreesurferStatsDataset.HBN,
    ]:
        df = tabularize_all_stats_files(dataset=dataset)
        print(df.iloc[:, :10])
        is_bad, dupes, bad_names = detect_table_issues(df)
        if is_bad:
            print(dupes)
            raise ValueError("Dupes already in base table")

        df = fill_missing_rois(df)
        is_bad, dupes, bad_names = detect_table_issues(df)
        if is_bad:
            print(dupes)
            raise ValueError("Dupes in filled table")

        df = add_bilateral_stats(df)
        is_bad, dupes, bad_names = detect_table_issues(df)
        if is_bad:
            print(dupes)
            raise ValueError("Dupes in bilateral stats")
        df = compute_cmcs(df)
        print(df)
