from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import re
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Index

from src.constants import MEMORY
from src.enumerables import FreesurferStatsDataset
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
    final_renames = {
        "CIS_P_Score": "CIS_P_Total",
        "ICU_P_Score": "ICU_P_Total",
        "NLES_SR_TotalOccurance": "NLES_SR_TotalOccurrence",
    }
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
    if colname in final_renames:
        colname = final_renames[colname]

    return colname


def rename_data_dict(dd: DataFrame) -> DataFrame:
    dd = dd.copy()
    celf_idx = dd.name.str.contains("CELF")
    celf_full = dd.scale_name.str.contains("Full")
    celf_8_9_idx = celf_idx & dd.scale_name.str.contains("5-8") & celf_full
    celf_9_21_idx = celf_idx & dd.scale_name.str.contains("9-21") & celf_full
    icu_p_idx = dd.name.str.contains("ICU") & dd.scale_name.str.contains("Parent")
    rbs_idx = dd.name.str.contains("Score_") & dd.scale_name.str.contains("RBS")

    dd.loc[celf_8_9_idx, "name"] = dd.loc[celf_8_9_idx, "name"].apply(
        lambda s: f"CELF_Full_5to8_{s}"
    )
    dd.loc[celf_9_21_idx, "name"] = dd.loc[celf_9_21_idx, "name"].apply(
        lambda s: f"CELF_Full_9to21_{s}"
    )
    dd.loc[icu_p_idx, "name"] = dd.loc[icu_p_idx, "name"].apply(
        lambda s: s.replace("ICU_", "ICU_P_")
    )
    dd.loc[rbs_idx, "name"] = dd.loc[rbs_idx, "name"].apply(lambda s: f"RBS_{s}")
    dd.loc[dd["name"] == "DTS_total"] = "DTS_Total"

    # fix more dumbness
    dumb = {
        "numerical": "numeric",
        "numeirc": "numeric",
        "decimal": "numeric",
        "character": "text",
        "DTS_Total": "numeric",
    }
    dd.loc[:, "type"] = dd["type"].str.lower()
    dd.loc[:, "type"] = dd["type"].apply(lambda x: dumb[x] if x in dumb else x)
    return dd


def remove_cols(df: DataFrame, dd: DataFrame) -> DataFrame:
    df = df.copy()
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
        "_Desc",  # large text field
        "_Invalid",  # text field, just explains NA
        "_Reason",  # text field, just explains NA
        "_reason",  # text field, just explains NA
        "_text",  # large text field
        "ColorVision",
        "ConsensusDx",  # mostly NaN, unknown meaning
        "DailyMeds",  # unrelated to MRI
        "DigitSpan",  #
        "DMDD",  # ?
        "DrugScreen",
        "EEG",  # EEG book-keeping
        "Incomplete",  # text field, just explains NA
        "MDD",  # ?
        "MRI",  # MRI book-keeping
        "NIDA",  # ?
        "Panic",  # ?
        "PBQ",  # pregnancy and birth questionnare
        "PhenX",  # questions about perception of neighbourhood and school
        "Physical",  # transient, unrelated to MRI
        "Pregnancy",
        "PreInt",  # pre-screening interview
        "Quotient",  # ?
        "RANRAS",  # ?
        "SocAnx",
        "SRS_P",  # all missing
        "SympChck",  # transient symptoms and symptoms
        "Tanner",  # physical sexual development
        "WHODAS",  # physical disability
        "YFAS",  # food addiction
        *pre_ints,  # no definition
    ]
    missing_scales = [
        "Pegboard",
        "Barratt",
        "CDI",
    ]
    individual_missing_cols = [
        "YSR_Stduy",
        "Basic_Demos_Commercial_Use",
        "Basic_Demos_Release_Number",
        "ICU_P_Callous",  # we factor reduce ourselves
        "ICU_P_Total",  # we factor reduce ourselves
        "ICU_P_Uncaring",  # we factor reduce ourselves
        "ICU_P_Unemotional",  # we factor reduce ourselves
        "PAQ_C_09_Avg",  # average of one set of qs...
        # NaN-related drops
        "CBCL_Pre_100a",  # way too many NaN
        "CBCL_Pre_100b",  # way too many NaN
        "CBCL_Pre_100c",  # way too many NaN
        "NLES_SR_TotalEvents",  # many NaN, should be re-derived as factor
        "NLES_SR_TotalOccurrence",  # many NaN, should be re-derived as factor
        "NLES_SR_Upset_Avg",  # many NaN, should be re-derived as factor
        "NLES_SR_Upset_Total",  # many NaN, should be re-derived as factor
        "TRF_113a",  # way too many Nan
        "TRF_113b",  # way too many Nan
        "TRF_113c",  # way too many Nan
    ]
    undefined = str("|".join(no_dict_cols + missing_scales + non_pred_cols))
    drops = df.filter(regex=undefined).columns
    df = df.drop(columns=drops)
    df = df.drop(columns=individual_missing_cols)
    # Only cols "missing" in data dictionary is now
    # ['Basic_Demos_Age', 'Basic_Demos_Sex', 'RBS_Total']
    # missing = sorted(set(df.columns.to_list()).difference(dd["name"].values))
    # print(missing)
    # print(len(missing))
    return df


@MEMORY.cache
def load_basic_cleaned() -> DataFrame:
    path = FreesurferStatsDataset.HBN.phenotypic_file()
    df = pd.read_csv(path, dtype=str, na_values=["NaN", "."])
    sids = df["Identifiers"].str.replace(",assessment", "").astype(str)
    df.index = Index(data=sids, name="sid").astype(str)
    df.drop(columns="Identifiers", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.rename(columns=dedup_colname)

    dd = pd.read_parquet(FreesurferStatsDataset.HBN.data_dictionary())
    dd = rename_data_dict(dd)
    df = remove_cols(df=df, dd=dd)
    df = df.astype(float)
    return df


def load_hbn_pheno(clear_cache: bool = False) -> DataFrame:
    if clear_cache:
        load_basic_cleaned.clear(warn=True)
    df = load_basic_cleaned()

    return df


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
    # load_hbn_pheno(clear_cache=True)
    cmc = compute_CMC_table(dataset=FreesurferStatsDataset.HBN, use_cached=False)
    # load_hbn_complete()
    print()
