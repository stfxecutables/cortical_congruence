from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Index
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.preprocessing import LabelEncoder

from src.constants import ABIDE_II_ENCODING
from src.enumerables import FreesurferStatsDataset, Tag
from src.munging.fs_stats.tabularize import compute_CMC_table, to_wide_subject_table


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
    dsm_labels = {
        "control": 0.0,
        "autism": 1.0,
        "aspergers": 2.0,
        "PDD-NOS": 3.0,
        "aspergers-or-PDD-NOS": 4.0,
        np.nan: np.nan,
    }
    dsm_spectrum = {
        "control": 0.0,
        "autism": 1.0,
        "aspergers": 1.0,
        "PDD-NOS": 1.0,
        "aspergers-or-PDD-NOS": 1.0,
        np.nan: np.nan,
    }
    dx = {
        1: 1,  # 1 in their table is "autism",
        2: 0,  # 2 in their table is "control"
    }
    sex = {
        1: 1,  # 1 in their table is Male
        2: 0,  # 2 in their table is Female
    }

    path = FreesurferStatsDataset.ABIDE_I.phenotypic_file()
    df = pd.read_csv(path)
    # non_null = df.columns[(df.isnull().sum() < 300)]
    # Non-null columns are below:
    # [
    #     "SITE_ID",
    #     "SUB_ID",
    #     "DX_GROUP",
    #     "DSM_IV_TR",
    #     "AGE_AT_SCAN",
    #     "SEX",
    #     "FIQ",
    #     "VIQ",
    #     "PIQ",
    #     "FIQ_TEST_TYPE",
    #     "VIQ_TEST_TYPE",
    #     "PIQ_TEST_TYPE",
    #     "CURRENT_MED_STATUS",
    #     "EYE_STATUS_AT_SCAN",
    # ]
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
    }
    renames = {
        col: f"{Tag.combine(tags)}__{col}" for col, tags in list(keep_cols.items())[1:]
    }

    df["autism"] = df["autism"].apply(lambda x: dx[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_diagnoses[x])
    df["dsm_iv_spectrum"] = df["dsm_iv"].apply(lambda x: dsm_spectrum[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_labels[x])
    df["sex"] = df["sex"].apply(lambda x: sex[x])
    df["fiq"].replace(-9999.0, np.nan, inplace=True)
    df["viq"].replace(-9999.0, np.nan, inplace=True)
    df["piq"].replace(-9999.0, np.nan, inplace=True)
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


@lru_cache
def load_abide_ii_pheno() -> DataFrame:
    dsm_iv_diagnoses = {
        0: "control",
        1: "autism",
        2: "aspergers",
        3: "PDD-NOS",
        -9999: np.nan,
    }
    dsm_iv_labels = {
        "control": 0.0,
        "autism": 1.0,
        "aspergers": 2.0,
        "PDD-NOS": 3.0,
        np.nan: np.nan,
    }
    dsm5_diagnoses = {
        "control": 0.0,
        "autism": 1.0,
        "aspergers": 1.0,
        "PDD-NOS": 1.0,
        np.nan: np.nan,
    }
    dx = {
        1: 1,  # 1 in their table is "autism",
        2: 0,  # 2 in their table is "control"
    }
    sex = {
        1: 1,  # 1 in their table is Male
        2: 0,  # 2 in their table is Female
    }
    keep_cols = ["sid", "site", "age", "sex", "autism", "dsm_iv", "fiq", "viq", "piq"]

    path = FreesurferStatsDataset.ABIDE_II.phenotypic_file()
    df = pd.read_csv(path, encoding=ABIDE_II_ENCODING)
    # df.columns[(df.isnull().sum() < 300)]
    # [
    #     "SITE_ID",
    #     "SUB_ID",
    #     "DX_GROUP",
    #     "AGE_AT_SCAN ",
    #     "SEX",
    #     "HANDEDNESS_CATEGORY",
    #     "FIQ",
    #     "PIQ",
    #     "FIQ_TEST_TYPE",
    #     "PIQ_TEST_TYPE",
    #     "CURRENT_MED_STATUS",
    #     "CURRENT_MEDICATION_NAME",
    #     "EYE_STATUS_AT_SCAN",
    # ]
    df.rename(
        columns={
            "SUB_ID": "sid",
            "SITE_ID": "site",
            "DX_GROUP": "autism",
            "PDD_DSM_IV_TR": "dsm_iv",
            "AGE_AT_SCAN ": "age",  # note space!
            "SEX": "sex",
            "FIQ": "fiq",
            "VIQ": "viq",
            "PIQ": "piq",
        },
        inplace=True,
    )
    df.fillna(-9999, inplace=True)
    keep_cols = {
        "sid": [],
        # "site": [Tag.Info],
        "sex": [Tag.Demographics, Tag.Feature],
        "age": [Tag.Demographics, Tag.Feature],
        "autism": [Tag.Target, Tag.Classification],
        "dsm_iv": [Tag.Target, Tag.Multiclass],
        "dsm5_spectrum": [Tag.Target, Tag.Multiclass],
        "fiq": [Tag.Target, Tag.Regression],
        "viq": [Tag.Target, Tag.Regression],
        "piq": [Tag.Target, Tag.Regression],
    }
    renames = {
        col: f"{Tag.combine(tags)}__{col}" for col, tags in list(keep_cols.items())[1:]
    }

    df["site"] = df["site"].str.replace("ABIDEII-", "")
    df["autism"] = df["autism"].apply(lambda x: dx[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_iv_diagnoses[x])
    df["dsm5_spectrum"] = df["dsm_iv"].apply(lambda x: dsm5_diagnoses[x])
    df["dsm_iv"] = df["dsm_iv"].apply(lambda x: dsm_iv_labels[x])
    df["sex"] = df["sex"].apply(lambda x: sex[x])
    df["fiq"].replace(-9999.0, np.nan, inplace=True)
    df["viq"].replace(-9999.0, np.nan, inplace=True)
    df["piq"].replace(-9999.0, np.nan, inplace=True)
    df = df[list(keep_cols.keys())].copy()
    df.rename(columns=renames, inplace=True)
    df.index = Index(name="sid", data=df["sid"].astype(str))
    df.drop(columns="sid", inplace=True)
    df.columns.name = "feature"
    return df
    # df = df.loc[:, keep_cols].copy()
    # df = df.rename(columns=lambda s: f"TARGET__{s}" if s in target_cols else s)
    # df = df.rename(columns=lambda s: f"DEMO__{s}" if s in demo_cols else s)
    # return df


@lru_cache
def load_adhd200_pheno() -> DataFrame:
    renames = {
        "ScanDir ID": "sid",
        "Site": "site",
        "Gender": "sex",
        "Age": "age",
        "DX": "diagnosis",
        "ADHD Measure": "adhd_scale",
        "ADHD Index": "adhd_score",
        "Inattentive": "adhd_score_inattentive",
        "Hyper/Impulsive": "adhd_score_hyper",
        "IQ Measure": "iq_scale",
        "Full4 IQ": "fiq",
        "Verbal IQ": "viq",
        "Performance IQ": "piq",
    }
    binary_dx = {
        "0": 0.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
        "pending": np.nan,
    }

    path = FreesurferStatsDataset.ADHD_200.phenotypic_file()
    df = pd.read_csv(path, sep="\t")
    df.rename(columns=renames, inplace=True)
    # print(df.columns[df.isnull().sum() < 400].to_list())
    # 'ScanDir ID', 'Site',
    # 'Gender', 'Age', 'Handedness',
    # 'DX', 'ADHD Measure', 'ADHD Index', 'Inattentive', 'Hyper/Impulsive',
    # 'IQ Measure', 'Verbal IQ', 'Performance IQ', 'Full4 IQ',
    # 'Med Status',
    # 'QC_Athena', 'QC_NIAK'

    """
    There are unfortunately two different ADHD scales used here and rolled into
    one column:

        1 ADHD Rating Scale IV (ADHD-RS)
        2 Conners’ Parent Rating Scale-Revised, Long version (CPRS-LV)
        3 Connors’ Rating Scale-3rd Edition [None in table]

    adhd = (
        df.filter(regex="adhd")
          .applymap(lambda x: np.nan if x in ["pending", "-999", -999] else x)
          .astype(float)
          .groupby("adhd_scale")
          .describe().T
    )
    adhd_scale                      1.0    2.0   3.0
    adhd_score             count  222.0  298.0   0.0
                           mean    38.3   57.8   NaN
                           std     12.7   15.0   NaN
                           min     18.0   40.0   NaN
                           25%     27.2   43.0   NaN
                           50%     36.0   55.5   NaN
                           75%     48.0   71.0   NaN
                           max     68.0   99.0   NaN

    adhd_score_inattentive count  222.0  298.0  79.0
                           mean    21.0   57.5  59.1
                           std      7.2   14.8  14.8
                           min      9.0   40.0  40.0
                           25%     14.6   43.2  45.0
                           50%     20.0   55.0  58.0
                           75%     27.8   69.8  70.0
                           max     36.0   90.0  90.0

    adhd_score_hyper       count  222.0  298.0  79.0
                           mean    17.4   57.1  57.4
                           std      6.8   14.3  15.9
                           min      9.0   41.0  40.0
                           25%     12.0   44.0  44.0
                           50%     16.0   52.0  51.0
                           75%     21.0   68.0  69.0
                           max     36.0   90.0  90.0

    Clearly, the ADHD-RS and Conners' scales are different instruments - to
    combine them into a single predictable ADHD score, we could do one of some
    options:

    1. convert each scale scores to z-scores and then report "adhd_z_score" as
       a new total / target
    2. convert each scale to percentiles "adhd_perc"
    3. construct a new scale which is comprised of the inattentive and hyper
       features, either as a sum of z-scores, or as a factor reduction
    """
    adhd = (
        df.filter(regex="adhd")
        .applymap(lambda x: np.nan if x in ["pending", "-999", -999, "NaN", "nan"] else x)
        .astype(float)
    )
    adhd["adhd_scale"] = adhd["adhd_scale"].astype(str)
    stats: DataFrame = adhd.groupby("adhd_scale").describe()
    sds = stats.loc[  # type: ignore
        ["1.0", "2.0", "3.0", "nan"],
        (["adhd_score", "adhd_score_inattentive", "adhd_score_hyper"], "std"),
    ].T.droplevel(1)
    means = stats.loc[  # type: ignore
        ["1.0", "2.0", "3.0", "nan"],
        (["adhd_score", "adhd_score_inattentive", "adhd_score_hyper"], "mean"),
    ].T.droplevel(1)

    """
    Above `sds` looks like:

        adhd_scale               1.0    2.0    3.0
        adhd_score             12.74  15.03    NaN
        adhd_score_inattentive  7.24  14.83  14.76
        adhd_score_hyper        6.76  14.29  15.87

    and above `means` looks like:

        adhd_scale               1.0   2.0   3.0
        adhd_score              38.3  57.8   NaN
        adhd_score_inattentive  21.0  57.5  59.1
        adhd_score_hyper        17.4  57.1  57.4

    """

    # Construct ADHD z-scores
    for raw_col in ["adhd_score", "adhd_score_inattentive", "adhd_score_hyper"]:
        z_col = raw_col.replace("_score", "_z")
        adhd[z_col] = adhd[raw_col].copy()
        for scale in ["1.0", "2.0", "3.0", "nan"]:
            idx = adhd["adhd_scale"] == scale
            centered = adhd.loc[idx, raw_col] - means.loc[raw_col, scale]
            adhd.loc[idx, z_col] = centered / sds.loc[raw_col, scale]
    adhd["adhd_z_sum"] = adhd["adhd_z_inattentive"] + adhd["adhd_z_hyper"]

    # now compute a factor reduction. Note the correlation between
    # "adhd_z_inattentive" and "adhd_z_hyper" is ~0.76 and the plot is without
    # clusters, however:
    #
    # adhd.adhd_score_hyper.corr(adhd.adhd_score_inattentive) = 0.93 due to
    # extreme clustering (like a Simpson's paradox, except the two clsuter
    # ellipsoids happen to have both axes parallel to each other, roughly.
    # So the z-score conversion largely removes this clustering and it probably
    # makes more sense to do the factor reduction on the z-scores.
    # NOTE: Using the mean to fill NA values maps the NA to 0 on the latent factor,
    # which has a mean of zero by construction anyway
    z_sub_scores = adhd[["adhd_z_inattentive", "adhd_z_hyper"]].copy()
    z_sub_scores_full = z_sub_scores.dropna()
    z_sub_scores_interp = z_sub_scores.fillna(z_sub_scores.mean())
    fitted = FA(1, svd_method="lapack", rotation="varimax").fit(z_sub_scores_full)
    adhd["adhd_factor"] = fitted.transform(z_sub_scores_interp).ravel()
    constructed_cols = [
        "adhd_z",
        "adhd_z_inattentive",
        "adhd_z_hyper",
        "adhd_z_sum",
        "adhd_factor",
    ]
    constructeds = adhd[constructed_cols].copy()
    df = pd.concat([df, constructeds], axis=1)
    # Diagnosis:
    #     0 Typically Developing Children
    #     1 ADHD-Combined
    #     2 ADHD-Hyperactive/Impulsive
    #     3 ADHD-Inattentive

    site = {
        1: "Peking University",
        2: "Bradley Hospital/Brown University",
        3: "Kennedy Krieger Institute",
        4: "NeuroIMAGE Sample",
        5: "New York University Child Study Center",
        6: "Oregon Health & Science University",
        7: "University of Pittsburgh",
        8: "Washington University in St. Louis",
    }

    df["site"] = df["site"].apply(lambda x: site[x])
    df["adhd_spectrum"] = df["diagnosis"].apply(lambda x: binary_dx[x])

    keep_cols = {
        "sid": [],
        # "site": [Tag.Info],
        # "Handedness": [Tag.Demographics, Tag.Feature],
        "sex": [Tag.Demographics, Tag.Feature],
        "age": [Tag.Demographics, Tag.Feature],
        # "dsm_iv": [Tag.Target, Tag.Classification],
        # "dsm5_spectrum": [Tag.Target, Tag.Classification],
        "diagnosis": [Tag.Target, Tag.Multiclass],
        "adhd_spectrum": [Tag.Target, Tag.Classification],
        "adhd_score": [Tag.Target, Tag.Regression],
        "adhd_score_inattentive": [Tag.Target, Tag.Regression],
        "adhd_score_hyper": [Tag.Target, Tag.Regression],
        "adhd_z": [Tag.Target, Tag.Regression, Tag.Constructed],
        "adhd_z_inattentive": [Tag.Target, Tag.Regression, Tag.Constructed],
        "adhd_z_hyper": [Tag.Target, Tag.Regression, Tag.Constructed],
        "adhd_z_sum": [Tag.Target, Tag.Regression, Tag.Constructed],
        "adhd_factor": [Tag.Target, Tag.Regression, Tag.Reduced],
        "fiq": [Tag.Target, Tag.Regression],
        "viq": [Tag.Target, Tag.Regression],
        "piq": [Tag.Target, Tag.Regression],
    }
    renames = {
        col: f"{Tag.combine(tags)}__{col}" for col, tags in list(keep_cols.items())[1:]
    }
    df = df.loc[:, list(keep_cols.keys())].copy()
    df.rename(columns=renames, inplace=True)
    df.index = Index(name="sid", data=df["sid"].astype(str))
    df.drop(columns="sid", inplace=True)
    df.columns.name = "feature"
    df = df.applymap(lambda x: np.nan if (x == "-999" or x == "pending") else x)
    targ_cols = df.filter(regex="TARGET").columns
    df.loc[:, targ_cols] = df[targ_cols].astype(float)
    return df


def load_abide_i_complete() -> DataFrame:
    pheno = load_abide_i_pheno()
    cmc = compute_CMC_table(dataset=FreesurferStatsDataset.ABIDE_I)
    wide = to_wide_subject_table(cmc)
    # right merges because pheno files are more complete
    # return pd.merge(wide, pheno, on="sid", how="right")
    # inner merges are all that make sense due to NaNs...
    complete = pd.merge(wide, pheno, on="sid", how="inner")
    return complete


def load_abide_ii_complete() -> DataFrame:
    pheno = load_abide_ii_pheno()
    cmc = compute_CMC_table(dataset=FreesurferStatsDataset.ABIDE_II)
    wide = to_wide_subject_table(cmc)
    # right merges because pheno files are more complete
    # return pd.merge(wide, pheno, on="sid", how="right")
    # inner merges are all that make sense due to NaNs...
    complete = pd.merge(wide, pheno, on="sid", how="inner")
    return complete


def load_adhd200_complete() -> DataFrame:
    pheno = load_adhd200_pheno()
    cmc = compute_CMC_table(dataset=FreesurferStatsDataset.ADHD_200)
    wide = to_wide_subject_table(cmc)
    # right merges because pheno files are more complete
    # return pd.merge(wide, pheno, on="sid", how="right")
    # inner merges are all that make sense due to NaNs...
    complete = pd.merge(wide, pheno, on="sid", how="inner")
    return complete


if __name__ == "__main__":
    load_abide_i_complete()
    load_abide_ii_complete()
    load_adhd200_complete()
