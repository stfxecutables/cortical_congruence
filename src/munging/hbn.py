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
    typo_fixes = {
        "CBCLpre": "CBCL_Pre",
        "CBCLPre": "CBCL_Pre",
        "CELF5_Meta": "",
        "TRF_Pre": "TRF_P",
        "SRS_Pre": "SRS_P",
    }
    final_removes = [
        "ESWAN_",
        "Vineland_",
        "CELF_Meta_",
    ]
    for typo, fix in typo_fixes.items():
        if typo in colname:
            colname = colname.replace(typo, fix)

    comma = colname.find(",")
    if comma < 0:
        return colname.replace(",", "_")
    label = colname[:comma]
    found = re.findall(label, colname)
    if len(found) > 1:
        colname = colname[comma + 1 :]
    colname = colname.replace(",", "_")
    for remove in final_removes:
        colname = colname.replace(remove, "")

    return colname


@lru_cache
def load_hbn_pheno() -> DataFrame:
    path = FreesurferStatsDataset.HBN.phenotypic_file()
    df = pd.read_csv(path, dtype=str, na_values=["NaN", "."])
    sids = df["Identifiers"].str.replace(",assessment", "").astype(str)
    df.index = Index(data=sids, name="sid")
    df.drop(columns="Identifiers", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.rename(columns=dedup_colname)
    # df.rename(columns=lambda s: s[s.find(",") + 1 :], inplace=True)

    dd = pd.read_parquet(FreesurferStatsDataset.HBN.data_dictionary())
    celf_idx = dd.name.str.contains("CELF")
    celf_full = dd.scale_name.str.contains("Full")
    celf_8_9_idx = celf_idx & dd.scale_name.str.contains("5-8") & celf_full
    celf_9_21_idx = celf_idx & dd.scale_name.str.contains("9-21") & celf_full
    dd.loc[celf_8_9_idx, "name"] = dd.loc[celf_8_9_idx, "name"].apply(
        lambda s: f"CELF_Full_5to8_{s}"
    )
    dd.loc[celf_9_21_idx, "name"] = dd.loc[celf_9_21_idx, "name"].apply(
        lambda s: f"CELF_Full_9to21_{s}"
    )
    pre_ints = dd[dd.scale_name.str.contains("Pre-Int")]["name"].to_list()
    no_dict_cols = [
        "Administration",
        "Data_entry",
        "EID",
        "START_DATE",
        "Season",
        "Site",
        "Study",
        "Year",
        "AUDIT",
        "Days_Baseline",
    ]
    non_pred_cols = [
        *pre_ints,
        "ColorVision",
        "ConsensusDx",  # mostly NaN, unknown meaning
        "DailyMeds",  # unrelated to MRI
        "DigitSpan",  #
        "DMDD",  # ?
        "DrugScreen",
        "EEG",  # EEG book-keeping
        "MRI",  # MRI book-keeping
        "NIDA",  # ?
        "Physical",  # transient, unrelated to MRI
        "Pregnancy",
        "PreInt",  # pre-screening interview
        "Quotient",  # ?
        "PhenX",  # questions about perception of neighbourhood and school
        "RANRAS",  # ?
        "SympChck",  # transient symptoms and symptoms
        "Tanner",  # physical sexual development
        "YFAS",  # food addiction
    ]
    missing_scales = [
        "Pegboard",
        "Barratt",
        "CDI",
    ]
    undefined = str("|".join(no_dict_cols + missing_scales + non_pred_cols))
    drops = df.filter(regex=undefined).columns
    df = df.drop(columns=drops)
    missing = sorted(set(df.columns.to_list()).difference(dd["name"].values))
    print(missing)
    print(len(missing))

    # len(set(df.rename(columns=dedup_colname).columns).intersection(dd.name))
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
