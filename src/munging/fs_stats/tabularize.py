from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tqdm.contrib.concurrent import process_map

from src.constants import ALL_STATSFILES, CACHED_RESULTS
from src.enumerables import FreesurferStatsDataset
from src.munging.fs_stats.parse import FreesurferStats


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
    vols = [vol for vol in vols if vol in other]
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
    """
    Make sure every right- and left- hemisphere ROI has an ROI on corresponding
    side, and that each subject has an entry for all left, right, and bilateral
    ROIs that appear in the data
    """
    df = df_stats.copy()
    is_corrupt, dupes, bad_names = detect_table_issues(df)
    if is_corrupt:
        print(dupes)
        raise ValueError("Duplicates in table. See above.")

    left = df[df["hemi"] == "left"]
    right = df[df["hemi"] == "right"]
    both = df[df["hemi"] == "both"]

    left_unq = left["StructName"].unique().tolist()
    right_unq = right["StructName"].unique().tolist()
    both_unq = both["StructName"].unique().tolist()

    # ensure any left- and right- ROIs don't have missing opposite / paired ROIs
    lrs = [l.replace("lh-", "rh-") for l in left_unq]  # noqa
    rls = [r.replace("rh-", "lh-") for r in right_unq]
    unq_names = sorted(set(both_unq + left_unq + right_unq + lrs + rls))

    df = df.groupby("sid").apply(_add_missing_rows(unq_names))
    df.index = range(len(df))  # type: ignore
    left = df[df["hemi"] == "left"]
    right = df[df["hemi"] == "right"]
    detect_table_symmetry(df)
    return df


def compute_cmcs(bilateral_df: DataFrame) -> DataFrame:
    df = bilateral_df.copy()
    detect_table_symmetry(df)

    df["CMC"] = df["GrayVol"] / df["pseudoVolume"]
    # NOTE: It seems that pseudoVolume is undefined exactly when Volume_mm3
    # is defined, so we can't ever compute CMC_vox...
    # df["CMC_vox"] = df["Volume_mm3"] / df["pseudoVolume"]
    # df["CMC_vox"].replace(np.inf, np.nan, inplace=True)

    left = (
        df[df["hemi"] == "left"]
        .sort_values(by=["sid", "StructName"])
        .reset_index(drop=True)
    )
    right = (
        df[df["hemi"] == "right"]
        .sort_values(by=["sid", "StructName"])
        .reset_index(drop=True)
    )

    bnames = left["StructName"].str.replace("lh-", "bh-")
    both = (
        df[df["hemi"] == "both"]
        .sort_values(by=["sid", "StructName"])
        .reset_index(drop=True)
    )
    both_lr = both[both["StructName"].isin(bnames)].sort_values(by=["sid", "StructName"])
    both_only = both[~both["StructName"].isin(bnames)].sort_values(
        by=["sid", "StructName"]
    )

    cmc_asym = left["CMC"] - right["CMC"]
    cmc_asym_abs = cmc_asym.abs()
    cmc_asym_div = left["CMC"] / right["CMC"]

    cmc_asym.name = "CMC_asym"
    cmc_asym_abs.name = "CMC_asym_abs"
    cmc_asym_div.name = "CMC_asym_div"

    left["CMC__asym"] = cmc_asym
    left["CMC__asym_abs"] = cmc_asym_abs
    left["CMC__asym_div"] = cmc_asym_div

    right["CMC__asym"] = cmc_asym
    right["CMC__asym_abs"] = cmc_asym_abs
    right["CMC__asym_div"] = cmc_asym_div

    both_lr["CMC__asym"] = cmc_asym
    both_lr["CMC__asym_abs"] = cmc_asym_abs
    both_lr["CMC__asym_div"] = cmc_asym_div

    both_only["CMC__asym"] = np.nan
    both_only["CMC__asym_abs"] = np.nan
    both_only["CMC__asym_div"] = np.nan

    df_cmc = pd.concat([left, right, both_lr, both_only], axis=0, ignore_index=True)

    return df_cmc


def compute_CMC_table(dataset: FreesurferStatsDataset) -> DataFrame:
    if dataset not in [
        FreesurferStatsDataset.ABIDE_I,
        FreesurferStatsDataset.ABIDE_II,
        FreesurferStatsDataset.ADHD_200,
        FreesurferStatsDataset.HBN,
    ]:
        raise NotImplementedError()

    out = CACHED_RESULTS / f"{dataset.value}__CMC_stats.parquet"
    if out.exists():
        return pd.read_parquet(out)

    df = tabularize_all_stats_files(dataset=dataset)
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
    print(df.iloc[:, :10])
    df.to_parquet(out)
    print(f"Cached CMC stats table to {out}")
    return df


def to_wide_subject_table(df_cmc: DataFrame) -> DataFrame:
    df = df_cmc.copy()
    meta_cols = [
        "sid",
        "sname",
        "hemi",
        "SegId",
        "Volume_mm3",
        "GrayVol",
        "SurfArea",
        "ThickAvg",
        "pseudoVolume",
    ]
    drop_cols = ["Struct", "parent", "fname", "SegId", "hemi", "sname"]

    meta_cols = [c for c in meta_cols if c in df.columns]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df_long = (
        df.drop(columns=drop_cols)
        .rename(
            columns={
                "ThickAvg": "FS__THICK_",
                "SurfArea": "FS__AREA_",
                "GrayVol": "FS__VOL_",
                "pseudoVolume": "FS__PVOL_",
            }
        )
        .rename(columns=lambda s: f"{s}__" if "CMC" in s else s)
        .melt(id_vars=["sid", "StructName"], var_name="metric")
    )
    df_long["feature"] = df_long["metric"] + df_long["StructName"]
    df_long.drop(columns=["StructName", "metric"], inplace=True)
    df_wide = df_long.pivot(index="sid", columns="feature")["value"].copy()
    if isinstance(df_wide, Series):
        df_wide = df_wide.to_frame()
    return df_wide
