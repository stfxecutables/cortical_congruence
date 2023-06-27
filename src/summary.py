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
import json
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
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
from typing_extensions import Literal
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing import Iterable


from src.enumerables import ROISource
from src.constants import (
    CACHED_RESULTS,
    BASE_STATFILES,
    DATA,
    ALL_STATSFILES,
    PIAL_STATFILES,
    DKT_STATFILES,
)
from src.fs_stats import FreesurferStats


@dataclass
class ROINames:
    source: ROISource
    names: list[str]
    has_SurfArea: bool


def _collect_stats_names(stat: Path) -> tuple[list[str], list[str]]:
    fs = FreesurferStats.from_statsfile(stat)
    df = fs.to_subject_table()
    names = df["StructName"].to_list()
    # Check if we have a "SurfArea column in the stats file"
    if fs.has_area:
        return names, []
    return [], names


def _collect_names(stats: list[Path]) -> tuple[set, set]:
    all_area_names, all_no_area_names = list(
        zip(
            *process_map(
                _collect_stats_names, stats, desc="Collecting column names", chunksize=1
            )
        )
    )
    area_names, no_area_names = set(), set()
    for names in all_area_names:
        area_names.update(names)
    for names in all_no_area_names:
        no_area_names.update(names)
    return area_names, no_area_names


def _to_lh_rh_only(names: Iterable[str]) -> list[str]:
    return sorted([n for n in names if re.match("[lr]h-", n) is not None])


def _strip_lh_rh(names: Iterable[str]) -> list[str]:
    return sorted([n.replace("lh-", "").replace("rh-", "") for n in names])


def collect_struct_names(
    source: ROISource,
    lateral_only: bool = False,
    strip_lh_rh: bool = False,
) -> tuple[ROINames, ROINames]:
    s = source.value
    l = "_lateral-only" if lateral_only else ""
    n = "_no-lh-rh" if strip_lh_rh else ""
    out = CACHED_RESULTS / f"roi_names__{s}{l}{n}.json"
    out_no_area = CACHED_RESULTS / f"roi_names__{s}{l}{n}_no-area.json"
    if out.exists() and out_no_area.exists():
        with open(out, "r") as handle:
            area_names: Iterable[str] = json.load(handle)
        with open(out_no_area, "r") as handle:
            no_area_names: Iterable[str] = json.load(handle)
        return (
            ROINames(source=source, names=sorted(area_names), has_SurfArea=True),
            ROINames(source=source, names=sorted(no_area_names), has_SurfArea=False),
        )

    root = DATA
    all_stats = root.rglob("*.stats")
    all_stats = sorted(list(filter(lambda p: p.name in ALL_STATSFILES, all_stats)))
    if source is ROISource.Base:
        stats = list(filter(lambda p: p.name in BASE_STATFILES, all_stats))
    elif source is ROISource.Pial:
        stats = list(filter(lambda p: p.name in PIAL_STATFILES, all_stats))
    elif source is ROISource.DKT:
        stats = list(filter(lambda p: p.name in DKT_STATFILES, all_stats))
    else:
        raise ValueError(f"Not a valid source of ROIs: {source}")

    area_names, no_area_names = _collect_names(stats)

    if lateral_only:
        # remove useless structures with no info to compute CMC
        area_names = _to_lh_rh_only(area_names)
        no_area_names = _to_lh_rh_only(no_area_names)

    if strip_lh_rh:
        area_names = _strip_lh_rh(area_names)
        no_area_names = _strip_lh_rh(no_area_names)

    area_names = list(sorted(area_names))
    no_area_names = list(sorted(no_area_names))
    with open(out, "w") as handle:
        json.dump(area_names, handle)
    print(f"Wrote names to {out}")
    with open(out_no_area, "w") as handle:
        json.dump(no_area_names, handle)

    # Note currently, for Pial and DKT sources, the no_area names are empty lists
    return (
        ROINames(source=source, names=area_names, has_SurfArea=True),
        ROINames(source=source, names=no_area_names, has_SurfArea=False),
    )


if __name__ == "__main__":
    area, no_area = collect_struct_names(source=ROISource.Base)
    print("=" * 83)
    print("Base names:")
    print(area)
    print(no_area)

    area, no_area = collect_struct_names(source=ROISource.Pial)
    print("=" * 83)
    print("Pial names:")
    print(area)
    print(no_area)

    area, no_area = collect_struct_names(source=ROISource.DKT)
    print("=" * 83)
    print("DKTAtlas names:")
    print(area)
    print(no_area)
