from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import re
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from functools import reduce
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import ABIDE_I, ABIDE_II
from src.freesurfer import (
    bilateral_stats_to_row,
    collect_struct_names,
    compute_bilateral_stats,
    get_subject_meta,
    merge_stats,
)


def collect_names(root: Path) -> set:
    return collect_struct_names(root, abide2_extra=False)


def collect_names_extra(root: Path) -> set:
    return collect_struct_names(root, abide2_extra=True)


def collect_all_names() -> None:
    abide_i_roots = sorted(ABIDE_I.rglob("stats"))
    abide_ii_roots = sorted(
        filter(lambda s: not ("fsaverage" in str(s)), ABIDE_II.rglob("stats"))
    )

    names_i = process_map(
        collect_names,
        abide_i_roots,
        desc="Loading and parsing ABIDE-I StructNames",
    )
    names_ii = process_map(
        collect_names,
        abide_ii_roots,
        desc="Loading and parsing ABIDE-II StructNames",
    )
    names_ii_extra = process_map(
        collect_names_extra,
        abide_ii_roots,
        desc="Loading and parsing ABIDE-II extra StructNames",
    )
    names_i = [n for n in names_i if n is not None]
    names_ii = [n for n in names_ii if n is not None]
    names_ii_extra = [n for n in names_ii_extra if n is not None]

    sep = "=" * 80
    print(sep)
    print("ABIDE-I names")
    print(sep)
    print(sorted(set().union(*names_i)))

    print(sep)
    print("ABIDE-II names")
    print(sep)
    print(sorted(set().union(*names_ii)))

    print(sep)
    print("ABIDE-II extra names")
    print(sep)
    print(sorted(set().union(*names_ii_extra)))


def load_row_no_extra(root: Path) -> tuple[DataFrame, DataFrame] | None:
    try:
        abide = 2 if "ABIDE-II" in str(root) else 1
        df = merge_stats(root)
        if df is None:
            return None

        stats = compute_bilateral_stats(df, abide=abide, extra=False)
        meta = get_subject_meta(int(stats.iloc[0, 0]))  # type: ignore
        for col in meta.columns.to_list():
            stats.insert(2, col, meta[col].item())

        df = bilateral_stats_to_row(stats)
        return df, stats
    except Exception as e:
        traceback.print_exc()
        print(f"Got error processing stats files at {root}: {e}")
        return None


def load_row_extra(root: Path) -> tuple[DataFrame, DataFrame] | None:
    try:
        extra = "ABIDE-II" in str(root)
        abide = 2 if extra else 1
        df = merge_stats(root, abide2_extra=extra)
        if df is None:
            return None
        stats = compute_bilateral_stats(df, abide=abide, extra=extra)
        df = bilateral_stats_to_row(stats)
        return df, stats
    except Exception as e:
        traceback.print_exc()
        print(f"Got error processing stats files at {root}: {e}")
        return None


def make_separate_tables() -> None:
    abide_i_roots = sorted(ABIDE_I.rglob("stats"))
    abide_ii_roots = sorted(
        filter(lambda s: not ("fsaverage" in str(s)), ABIDE_II.rglob("stats"))
    )

    df_abide_iis = process_map(
        # load_row_no_extra,
        load_row_extra,
        abide_ii_roots,
        # abide_ii_roots[:2],
        desc="Loading and parsing ABIDE-II stats + extra",
    )

    df_abide_is = process_map(
        load_row_no_extra, abide_i_roots, desc="Loading and parsing ABIDE-I stats"
    )

    df_abide_i = pd.concat(
        [df for df in df_abide_is if df is not None], axis=0, ignore_index=True
    )
    df_abide_ii = pd.concat(
        [df for df in df_abide_iis if df is not None], axis=0, ignore_index=True
    )

    df_abide_i.to_json(ROOT / "abide_i_cmc.json")
    df_abide_i.to_parquet(ROOT / "abide_i_cmc.parquet")
    df_abide_i.to_csv(ROOT / "abide_i_cmc.csv")

    df_abide_ii.to_json(ROOT / "abide_ii_cmc_extra.json")
    df_abide_ii.to_parquet(ROOT / "abide_ii_cmc_extra.parquet")
    df_abide_ii.to_csv(ROOT / "abide_ii_cmc_extra.csv")


def make_combined_table() -> None:
    abide_i_roots = sorted(ABIDE_I.rglob("stats"))
    abide_ii_roots = sorted(ABIDE_II.rglob("stats"))

    df_abide_is = process_map(
        load_row_no_extra, abide_i_roots, desc="Loading and parsing ABIDE-I stats"
    )
    df_abide_iis = process_map(
        load_row_no_extra, abide_ii_roots, desc="Loading and parsing ABIDE-II stats"
    )
    df_abide_is = [df for df in df_abide_is if df is not None]
    df_abide_iis = [df for df in df_abide_iis if df is not None]

    df_is, stats_is = list(zip(*df_abide_is))
    df_iis, stats_iis = list(zip(*df_abide_is))

    df_i = pd.concat(df_is, axis=0, ignore_index=True)
    df_ii = pd.concat(df_iis, axis=0, ignore_index=True)
    stats_i = pd.concat(stats_is, axis=0, ignore_index=True)
    stats_ii = pd.concat(stats_iis, axis=0, ignore_index=True)

    df_i.insert(5, "abide", "I")
    df_ii.insert(5, "abide", "II")
    stats_i.insert(2, "abide", "I")
    stats_ii.insert(2, "abide", "II")

    df = pd.concat([df_i, df_ii], axis=0, ignore_index=True)
    df.to_json(ROOT / "abide_cmc_combined.json")
    df.to_parquet(ROOT / "abide_cmc_combined.parquet")
    df.to_csv(ROOT / "abide_cmc_combined.csv")

    stats = pd.concat([stats_i, stats_ii], axis=0, ignore_index=True)
    stats.to_json(ROOT / "abide_stats_combined.json")
    stats.to_parquet(ROOT / "abide_stats_combined.parquet")
    stats.to_csv(ROOT / "abide_stats_combined.csv")
    print(df)


if __name__ == "__main__":
    # collect_all_names()
    make_combined_table()
    # make_separate_tables()
