from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
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

DATA = ROOT / "data"
ABIDE_I = DATA / "ABIDE-I"
ABIDE_II = DATA / "ABIDE-II"
ABIDE_II_ENCODING = "iso8859_2"

ABIDE_I_PHENO = DATA / "ABIDE_I_Phenotypic_V1_0b.csv"
ABIDE_II_PHENO = DATA / "ABIDEII_Composite_Phenotypic.csv"

COMMON_MISSING_ROIS = [
    "transversetemporal",
    "parahippocampal",
    "pericalcarine",
    "cuneus",
]
RARE_MISSING_ROIS = [
    "entorhinal",
    "fusiform",
    "temporalpole",
    "posteriorcingulate",
    "isthmuscingulate",
    "rostralanteriorcingulate",
    "caudalanteriorcingulate",
    "medialorbitofrontal",
    "paracentral",
]
ALL_ROIS = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "transversetemporal",
]


@lru_cache
def load_abide_i_pheno() -> DataFrame:
    dsm_diagnoses = {
        0: "control",
        1: "autism",
        2: "aspergers",
        3: "PDD-NOS",
        4: "aspergers-or-PDD-NOS",
        -9999: np.nan,
    }
    dx = {
        1: 1,  # 1 in their table is "autism",
        2: 0,  # 2 in their table is "control"
    }
    sex = {
        1: 1,  # 1 in their table is Male
        2: 0,  # 2 in their table is Female
    }
    keep_cols = ["sid", "site", "autism", "dsm_iv", "age", "sex"]

    df = pd.read_csv(ABIDE_I_PHENO)
    df.rename(
        columns={
            "SUB_ID": "sid",
            "SITE_ID": "site",
            "DX_GROUP": "autism",
            "DSM_IV_TR": "dsm_iv",
            "AGE_AT_SCAN": "age",
            "SEX": "sex",
        },
        inplace=True,
    )

    df["autism"] = df["autism"].apply(lambda x: dx[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_diagnoses[x])
    df["sex"] = df["sex"].apply(lambda x: sex[x])

    return df.loc[:, keep_cols].copy()


@lru_cache
def load_abide_ii_pheno() -> DataFrame:
    dsm_iv_diagnoses = {
        0: "control",
        1: "autism",
        2: "aspergers",
        3: "PDD-NOS",
        -9999: np.nan,
    }
    dsm5_diagnoses = {
        0: "control",
        1: "ASD",
        -9999: np.nan,
    }
    dx = {
        1: 1,  # 1 in their table is "autism",
        2: 0,  # 2 in their table is "control"
    }
    sex = {
        1: 1,  # 1 in their table is Male
        2: 0,  # 2 in their table is Female
    }
    keep_cols = ["sid", "site", "autism", "dsm_iv", "age", "sex"]

    df = pd.read_csv(ABIDE_II_PHENO, encoding=ABIDE_II_ENCODING)
    df.rename(
        columns={
            "SUB_ID": "sid",
            "SITE_ID": "site",
            "DX_GROUP": "autism",
            "PDD_DSM_IV_TR": "dsm_iv",
            "AGE_AT_SCAN ": "age",  # note space!
            "SEX": "sex",
        },
        inplace=True,
    )
    df.fillna(-9999, inplace=True)

    df["site"] = df["site"].str.replace("ABIDEII-", "")
    df["autism"] = df["autism"].apply(lambda x: dx[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_iv_diagnoses[x])
    df["sex"] = df["sex"].apply(lambda x: sex[x])

    return df.loc[:, keep_cols].copy()


if __name__ == "__main__":
    print(load_abide_i_pheno())
    print(load_abide_ii_pheno())
