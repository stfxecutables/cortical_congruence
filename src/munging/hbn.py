from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Index
from sklearn.decomposition import FactorAnalysis as FA

from src.enumerables import FreesurferStatsDataset, Tag
from src.munging.fs_stats.tabularize import compute_CMC_table, to_wide_subject_table


def dedup_colname(colname: str) -> str:
    comma = colname.find(",")
    if comma < 0:
        return colname.replace(",", "_")
    label = colname[:comma]
    found = re.findall(label, colname)
    if len(found) > 1:
        return colname[comma + 1 :]
    return colname.replace(",", "_")


@lru_cache
def load_hbn_pheno() -> DataFrame:
    path = FreesurferStatsDataset.HBN.phenotypic_file()
    df = pd.read_csv(path, dtype=str, na_values=["NaN", "."])
    sids = df["Identifiers"].str.replace(",assessment", "").astype(str)
    df.index = Index(data=sids, name="sid")
    df.drop(columns="Identifiers", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    # df.rename(columns=lambda s: s[s.find(",") + 1 :], inplace=True)

    dd = pd.read_parquet(FreesurferStatsDataset.HBN.data_dictionary())

    return df
    df.rename(
        columns={
            "SUB_ID": "sid",
            "SITE_ID": "site",
            "DX_GROUP": "autism",
            "DSM_IV_TR": "dsm_iv",
            "AGE_AT_SCAN": "age",
            "SEX": "sex",
            "FIQ": "fiq",
            "VIQ": "viq",
            "PIQ": "piq",
        },
        inplace=True,
    )
    keep_cols = {
        "sid": [],
        # "site": [Tag.Info],
        "sex": [Tag.Demographics, Tag.Feature],
        "age": [Tag.Demographics, Tag.Feature],
        "autism": [Tag.Target, Tag.Classification],
        "dsm_iv": [Tag.Target, Tag.Multiclass],
        "dsm_iv_spectrum": [Tag.Target, Tag.Classification],
        "fiq": [Tag.Target, Tag.Regression],
        "viq": [Tag.Target, Tag.Regression],
        "piq": [Tag.Target, Tag.Regression],
        "int_g_like": [Tag.Target, Tag.Regression],
    }
    renames = {
        col: f"{Tag.combine(tags)}__{col}" for col, tags in list(keep_cols.items())[1:]
    }

    df["sex"] = df["sex"].apply(lambda x: sex[x])
    df["fiq"].replace(-9999.0, np.nan, inplace=True)
    df["viq"].replace(-9999.0, np.nan, inplace=True)
    df["piq"].replace(-9999.0, np.nan, inplace=True)
    iq = df.filter(regex="fiq|viq|piq")
    df["int_g_like"] = FA(n_components=1, rotation="varimax").fit_transform(
        iq.fillna(iq.mean())
    )

    df = df.loc[:, list(keep_cols.keys())].copy()
    df.rename(columns=renames, inplace=True)
    df.index = Index(name="sid", data=df["sid"].astype(str))
    df.drop(columns="sid", inplace=True)
    df.columns.name = "feature"
    return df

    # df = df.loc[:, keep_cols].copy()
    # df = df.rename(columns=lambda s: f"TARGET__{s}" if s in target_cols else s)
    # df = df.rename(columns=lambda s: f"DEMO__{s}" if s in demo_cols else s)
    # return df


def load_hbn_complete() -> DataFrame:
    pheno = load_hbn_pheno()
    cmc = compute_CMC_table(dataset=FreesurferStatsDataset.HBN)
    wide = to_wide_subject_table(cmc)
    # right merges because pheno files are more complete
    # return pd.merge(wide, pheno, on="sid", how="right")
    # inner merges are all that make sense due to NaNs...
    complete = pd.merge(wide, pheno, on="sid", how="inner")
    return complete


if __name__ == "__main__":
    load_hbn_complete()
