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

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import FactorAnalysis as FA
from tqdm import tqdm

from src.constants import HCP_FEATURE_INFO, MEMORY
from src.munging.clustering import Cluster


@MEMORY.cache
def reduce_HCP_clusters(data: DataFrame, clusters: list[Cluster]) -> DataFrame:
    mappings = {
        "gambling_task_perc_larger": "gambling_perf",
        "mars_log_score": "mars",
        "language_task_median_rt": "language_rt",
        "psqi_comp1": "psqi_score",
        "relational_task_median_rt": "relational_rt",
        "emotion_task_median_rt": "emotion_rt",
        "emotion_task_acc": "emotion_perf",
        "pmat24_a_cr": "p_matrices",
        "wm_task_0bk_face_acc": "wm_face_acc",
        "social_task_median_rt_random": "social_rt",
        "social_task_perc_random": "social_random_perf",
        "language_task_acc": "language_perf",
        "social_task_perc_tom": "social_tom_perf",
        "gambling_task_median_rt_larger": "gambling_rt",
        "readeng_unadj": "int_g_like",
        "wm_task_0bk_face_acc": "wm_perf",
        "fearaffect_unadj": "neg_emotionality",
        "sadness_unadj": "neg_emotionality",
        "wm_task_0bk_body_median_rt": "wm_rt",
    }
    for cluster in clusters:
        if not (cluster.name in mappings):
            raise ValueError(
                f"Missing a cluster name for cluster with main feature: {cluster.name}"
            )

    reductions = []
    for cluster in tqdm(clusters, desc="Factor reducing..."):
        name = mappings[cluster.name]
        df = cluster.data
        feat_names = sorted(set(df.x.to_list() + df.y.to_list()))
        feats = data[feat_names]
        fa = FA(n_components=1, rotation="varimax")
        x = fa.fit_transform(feats.fillna(feats.median()))
        reductions.append(DataFrame(data=x, index=data.index, columns=[name]))
    df = pd.concat(reductions, axis=1)
    df.rename(columns=lambda s: f"target__{s}")
    return df


def get_available_features(df: DataFrame, extra: DataFrame) -> list[str]:
    feats = extra["columnHeader"].str.lower()
    extra = extra.rename(columns=str.lower)
    extra["columnheader"] = extra["columnheader"].str.lower()
    available = list(set(feats).intersection(df.columns))
    return available


def remove_uncontroversial_features(df: DataFrame, extra: DataFrame) -> DataFrame:
    df = df.copy()
    available = get_available_features(df, extra)
    freesurfer = sorted(filter(lambda f: "fs_" in f, available))
    remain = DataFrame(
        data=sorted(set(available).difference(freesurfer)), columns=["feature"]
    )
    feature_info = pd.read_csv(HCP_FEATURE_INFO)

    remain = pd.merge(
        remain,
        feature_info.loc[:, ["feature", "kind", "cardinality"]],
        how="left",
        on="feature",
    )
    remain.index = remain["feature"]  # type: ignore
    # kinds are:
    #       book_keeping: HCP book-keeping (e.g. n scans compled, scanner id, etc)
    #       physio: biological, e.g. dexterity, strength, smell sensitivity
    #       emotion: psych scores involving primarily / clearly emotional
    #       cognition: metrics assessing primarily cognition (e.g. working mem)
    #       psych: other psych scales (e.g. self-efficacy, well-being, delay-discounting)
    #       sub_scale: is a single item / score that is summed / combined into a total
    #       timing: is a time or reaction time to response in some fMRI task
    #       social: loneliness, TOM, or other specifically social cognitive measures

    # Prediction of single-item Likert or sub-scale values is known to be impossible
    # and scientifically useless, so there is no point even checking these. Likewise,
    # book-keeping variables are specific to the HCP data and also useless, since NaNs
    # perform the same function in all cases.
    junk_features = remain.loc[remain["kind"].isin(["book_keeping", "sub_scale"])].index
    df = df.drop(columns=junk_features)
    remain = remain.drop(index=junk_features)
    # cardinalities are:
    #       bin: binary categorical
    #       reg: regression (continuous or large ordinal)
    #       ord: small ordinal
    #       cat: categorical
    #       id: categorical but meaningless to predict (i.e. assigned identifier)
    #
    # Note that at this point, the only ordinal remaining is "painintens_rawscore", which
    # is a 1-10 ordinal, and so can be treated as regression, for simplicity, unless we
    # count the age_range variable. So to remove that issue, we just convert age to a
    # 4-point ordinal, and treat it as a regression.

    # Many remaining columns are extremely low quality and still useless due
    # to missing data or constancy. We remove those here.
    df = df.loc[:, df.isna().sum() < 200]  # removes 7 features
    # fmt: off
    bad_cats = [
        "release", "acquisition", "dmri_3t_reconvrs", "fmri_3t_reconvrs",
        "mrsession_scanner_3t", "mrsession_scans_3t", "mrsession_label_3t",
        "psqi_bedtime", "psqi_getuptime", "neoraw_01", "neoraw_02", "neoraw_03",
        "neoraw_04", "neoraw_05", "neoraw_06", "neoraw_07", "neoraw_08",
        "neoraw_09", "neoraw_10", "neoraw_11", "neoraw_12", "neoraw_13",
        "neoraw_14", "neoraw_15", "neoraw_16", "neoraw_17", "neoraw_18",
        "neoraw_19", "neoraw_20", "neoraw_21", "neoraw_22", "neoraw_23",
        "neoraw_24", "neoraw_25", "neoraw_26", "neoraw_27", "neoraw_28",
        "neoraw_29", "neoraw_30", "neoraw_31", "neoraw_32", "neoraw_33",
        "neoraw_34", "neoraw_35", "neoraw_36", "neoraw_37", "neoraw_38",
        "neoraw_39", "neoraw_40", "neoraw_41", "neoraw_42", "neoraw_43",
        "neoraw_44", "neoraw_45", "neoraw_46", "neoraw_47", "neoraw_48",
        "neoraw_49", "neoraw_50", "neoraw_51", "neoraw_52", "neoraw_53",
        "neoraw_54", "neoraw_55", "neoraw_56", "neoraw_57", "neoraw_58",
        "neoraw_59", "neoraw_60",
    ]
    # fmt: on
    df = df.drop(columns=bad_cats, errors="ignore").copy()
    return df


def remove_highly_correlated_features(df: DataFrame) -> DataFrame:
    """
    Below can be verified with the code:

    ```py
    idx = np.triu(np.ones(corrs.shape, dtype=bool), k=1)  # unique correlations
    corrs = (
        corrs.where(idx)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    )
    corrs["r"] = corrs["r"].abs()
    corrs = corrs.sort_values(by="r", ascending=False)
    print(corrs.iloc[:50])
    ```

    All "ageadj" features remain extremely strongly correlated with the unadj
    originals, and so this adjustment is useless from a causal / predictive
    modeling standpoint. We just get rid of "ageadj" variables.

                       x                      y         r
             taste_unadj           taste_ageadj  0.996431
      cogtotalcomp_unadj    cogtotalcomp_ageadj  0.995131
            picseq_unadj          picseq_ageadj  0.992739
         endurance_unadj       endurance_ageadj  0.991998
         dexterity_unadj       dexterity_ageadj  0.991142
      cogearlycomp_unadj    cogearlycomp_ageadj  0.991109
           readeng_unadj         readeng_ageadj  0.989491
          listsort_unadj        listsort_ageadj  0.987422
    cogcrystalcomp_unadj  cogcrystalcomp_ageadj  0.985509
              odor_unadj            odor_ageadj  0.983783
         procspeed_unadj       procspeed_ageadj  0.982411
          strength_unadj        strength_ageadj  0.980581
          picvocab_unadj        picvocab_ageadj  0.977979
           flanker_unadj         flanker_ageadj  0.977893
      cogfluidcomp_unadj    cogfluidcomp_ageadj  0.973162
          cardsort_unadj        cardsort_ageadj  0.969539

    The "mars" columns are also nearly identical and so the log score is dropped:

        mars_log_score   mars_final  0.997427


    In addition some columns are exact linear functions of others, and so are
    dropped:

                                   x                                   y    r
    gambling_task_punish_perc_larger   gambling_task_punish_perc_smaller  1.0
    gambling_task_reward_perc_larger   gambling_task_reward_perc_smaller  1.0
                             scpt_tp                             scpt_fn  1.0
                             scpt_tn                             scpt_fp  1.0
           gambling_task_perc_larger          gambling_task_perc_smaller  1.0
                             scpt_tp                            scpt_sen  1.0
                             scpt_fn                            scpt_sen  1.0
                             scpt_tn                           scpt_spec  1.0
                             scpt_fp                           scpt_spec  1.0
    """
    cluster_drops = "|".join(
        [
            "gambling_task_perc_smaller",
            "gambling_task_reward_perc_smaller",
            "gambling_task_punish_perc_smaller",
            "scpt_fn",
            "scpt_tp",
            "scpt_tn",
            "scpt_fp",
        ]
    )
    drops = df.filter(regex=f"ageadj|mars_log_score|{cluster_drops}").columns
    return df.drop(columns=drops)


def fsrename(col: str) -> str:
    """takes column of form:

    fs_[l_transversetemporal]_[thck|area|vol]

    to

    FS__[thck|area|vol]__[l_transversetemporal]
    """
    if "fs_" not in col:
        return col
    result = re.search("(_thck|_area|_vol])", col)
    if result is None:
        return f"FS__{col}"
    label = result[1]
    col = col.replace("fs_l_", "lh-").replace("fs_r_", "rh-").replace(label, "")
    label = label[1:].upper()
    return f"FS__{label}_{col}"


def cleanup_HCP_phenotypic(df: DataFrame, extra: DataFrame) -> DataFrame:
    df = df.rename(
        columns={"Subject": "sid", "Gender": "sex", "Age": "age_range"}
    ).rename(columns=str.lower)
    sids = df["sid"].astype(str)
    df = df.drop(columns="sid")
    df["age_range"] = df["age_range"].apply(
        lambda age: {"22-25": 0.0, "26-30": 1.0, "31-35": 2.0, "36+": 3.0}[age]
    )
    df["sex"] = df["sex"].apply(lambda s: 0 if s == "F" else 1).astype(np.float64)
    df.rename(columns={"age_range": "age_class"}, inplace=True)

    df = remove_uncontroversial_features(df=df, extra=extra)
    df = remove_highly_correlated_features(df)
    available = get_available_features(df, extra)
    freesurfer = sorted(filter(lambda f: "fs_" in f, available))
    nonfs = list(set(available).difference(freesurfer))

    demo_cols = ["sex", "age_class"]
    always_keep = list(set(demo_cols + freesurfer + nonfs))
    df = df[always_keep].copy()
    df.rename(columns=fsrename, inplace=True)
    df.rename(columns={"sex": "DEMO__sex", "age_class": "DEMO__age_class"}, inplace=True)
    df.rename(
        columns=lambda col: f"TARGET__{col}" if ("__" not in col) else col, inplace=True
    )

    drops = []
    for col in df.filter(regex="TARGET__").columns:
        if df[col].dtype == "O":
            drops.append(col)
            continue
        if df[col].std() <= 0.0 or df[col].isna().all():
            drops.append(col)
    df = df.drop(columns=drops)

    order = [
        *df.filter(regex="DEMO__").columns.to_list(),
        *df.filter(regex="FS__").columns.to_list(),
        *df.filter(regex="TARGET__").columns.to_list(),
    ]
    df = df.loc[:, order].copy()
    df.index = sids  # type: ignore
    df.columns.name = "feature"
    # finally, many subjects are missing all FreeSurfer features. We drop them.
    keep_sids = df.filter(regex="FS__").dropna(axis=0, how="all").index
    df = df.loc[keep_sids].copy()
    return df
