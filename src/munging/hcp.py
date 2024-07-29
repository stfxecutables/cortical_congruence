from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import re
import sys
from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import FactorAnalysis as FA
from tqdm import tqdm

from src.constants import HCP_FEATURE_INFO, MEMORY
from src.enumerables import FreesurferStatsDataset
from src.munging.clustering import Cluster, get_cluster_corrs, reduce_CMC_clusters
from src.munging.fs_stats.tabularize import (
    add_bilateral_stats,
    compute_cmcs,
    to_wide_subject_table,
)

TABLES = ROOT / "tables"


class PhenotypicFocus(Enum):
    All = "all"
    Reduced = "reduced"
    Focused = "focused"

    def hcp_dict_file(self) -> Path:
        root = FreesurferStatsDataset.HCP.phenotypic_file().parent
        files = {
            PhenotypicFocus.All: root / "all_features.csv",
            PhenotypicFocus.Reduced: root / "features_of_interest.csv",
            PhenotypicFocus.Focused: root / "priority_features_of_interest.csv",
        }
        if self not in files:
            raise ValueError(f"Invalid {self.__class__.__name__}: {self}")
        return files[self]

    def desc(self) -> str:
        return {
            PhenotypicFocus.All: "all",
            PhenotypicFocus.Reduced: "plausible",
            PhenotypicFocus.Focused: "most sound",
        }[self]


@MEMORY.cache
def reduce_HCP_clusters(
    data: DataFrame, clusters: list[Cluster]
) -> tuple[DataFrame, list[DataFrame]]:
    mappings = {
        "gambling_task_perc_larger": "gambling_perf",
        "mars_log_score": "mars",
        "language_task_median_rt": "language_rt",
        "psqi_comp1": "psqi_latent",
        "relational_task_median_rt": "relational_rt",
        "emotion_task_median_rt": "emotion_rt",
        "emotion_task_acc": "emotion_perf",
        "pmat24_a_cr": "p_matrices",
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

    def shortname(cluster: Cluster) -> str:
        name = cluster.name.replace("CLUST__", "")
        name = name[: name.find("__")]
        return name

    for cluster in clusters:
        if not (shortname(cluster) in mappings):
            raise ValueError(
                f"Missing a cluster name for cluster with main feature: {cluster.name}"
            )

    reductions = []
    loadings = []
    for cluster in tqdm(clusters, desc="Factor reducing..."):
        name = mappings[shortname(cluster)]
        df = cluster.data
        feat_names = sorted(set(df.x.to_list() + df.y.to_list()))
        feats = data[feat_names]
        fa = FA(n_components=1, rotation="varimax")
        with pd.option_context("display.max_rows", 500):
            print(f"Filling NaN values for cluster: {name}")
            print(feats.isnull().sum())
        x = fa.fit_transform(feats.fillna(feats.mean()))
        reduced = DataFrame(data=x, index=data.index, columns=[name])
        loading = pd.concat([feats, reduced], axis=1).corr()[name].to_frame()
        loading["abs"] = loading[name].abs()
        loading = loading.sort_values(by="abs", ascending=False).drop(columns="abs")
        reductions.append(reduced)
        loadings.append(loading)
    df = pd.concat(reductions, axis=1)
    df.rename(columns=lambda s: f"target__{s}")
    return df, loadings


def get_available_features(df: DataFrame, extra: DataFrame) -> list[str]:
    feats = extra["columnHeader"].str.lower()
    extra = extra.rename(columns=str.lower)
    extra["columnheader"] = extra["columnheader"].str.lower()
    available = list(set(feats).intersection(df.columns))
    return available


def remove_uncontroversial_features(df: DataFrame, extra: DataFrame) -> DataFrame:
    # fmt: off
    non_freesurfer = [
        "angaffect_unadj", "angaggr_unadj", "anghostil_unadj", "cardsort_unadj",
        "cogcrystalcomp_unadj", "cogearlycomp_unadj", "cogfluidcomp_unadj",
        "cogtotalcomp_unadj", "ddisc_auc_200", "ddisc_auc_40k", "dexterity_unadj",
        "emotion_task_acc", "emotion_task_face_acc", "emotion_task_face_median_rt",
        "emotion_task_median_rt", "emotion_task_shape_acc", "emotion_task_shape_median_rt",
        "emotsupp_unadj", "endurance_unadj", "er40_cr", "er40_crt", "er40ang", "er40fear",
        "er40hap", "er40noe", "er40sad", "fearaffect_unadj", "fearsomat_unadj",
        "flanker_unadj", "friendship_unadj", "gaitspeed_comp",
        "gambling_task_median_rt_larger", "gambling_task_median_rt_smaller",
        "gambling_task_perc_larger", "gambling_task_perc_nlr",
        "gambling_task_punish_median_rt_larger", "gambling_task_punish_median_rt_smaller",
        "gambling_task_punish_perc_larger", "gambling_task_punish_perc_nlr",
        "gambling_task_reward_median_rt_larger", "gambling_task_reward_median_rt_smaller",
        "gambling_task_reward_perc_larger", "gambling_task_reward_perc_nlr",
        "instrusupp_unadj", "iwrd_rtc", "iwrd_tot", "language_task_acc",
        "language_task_math_acc", "language_task_math_avg_difficulty_level",
        "language_task_math_median_rt", "language_task_median_rt",
        "language_task_story_acc", "language_task_story_avg_difficulty_level",
        "language_task_story_median_rt", "lifesatisf_unadj", "listsort_unadj",
        "loneliness_unadj", "mars_errs", "mars_final", "meanpurp_unadj", "mmse_score",
        "neofac_a", "neofac_c", "neofac_e", "neofac_n", "neofac_o", "odor_unadj",
        "painintens_rawscore", "paininterf_tscore", "perchostil_unadj", "percreject_unadj",
        "percstress_unadj", "picseq_unadj", "picvocab_unadj", "pmat24_a_cr", "pmat24_a_rtcr",
        "pmat24_a_si", "posaffect_unadj", "procspeed_unadj", "psqi_comp1", "psqi_comp2",
        "psqi_comp3", "psqi_comp4", "psqi_comp5", "psqi_comp6", "psqi_comp7", "psqi_score",
        "readeng_unadj", "relational_task_acc", "relational_task_match_acc",
        "relational_task_match_median_rt", "relational_task_median_rt",
        "relational_task_rel_acc", "relational_task_rel_median_rt", "sadness_unadj",
        "scpt_lrnr", "scpt_sen", "scpt_spec", "selfeff_unadj",
        "social_task_median_rt_random", "social_task_median_rt_tom", "social_task_perc_nlr",
        "social_task_perc_random", "social_task_perc_tom", "social_task_perc_unsure",
        "social_task_random_median_rt_random", "social_task_random_perc_nlr",
        "social_task_random_perc_random", "social_task_random_perc_tom",
        "social_task_random_perc_unsure", "social_task_tom_median_rt_tom",
        "social_task_tom_perc_nlr", "social_task_tom_perc_random",
        "social_task_tom_perc_tom", "social_task_tom_perc_unsure", "strength_unadj",
        "taste_unadj", "vsplot_crte", "vsplot_off", "vsplot_tc", "wm_task_0bk_acc",
        "wm_task_0bk_body_acc", "wm_task_0bk_body_acc_nontarget",
        "wm_task_0bk_body_acc_target", "wm_task_0bk_body_median_rt",
        "wm_task_0bk_body_median_rt_nontarget", "wm_task_0bk_body_median_rt_target",
        "wm_task_0bk_face_acc", "wm_task_0bk_face_acc_nontarget",
        "wm_task_0bk_face_acc_target", "wm_task_0bk_face_median_rt",
        "wm_task_0bk_face_median_rt_nontarget", "wm_task_0bk_face_median_rt_target",
        "wm_task_0bk_median_rt", "wm_task_0bk_place_acc", "wm_task_0bk_place_acc_nontarget",
        "wm_task_0bk_place_acc_target", "wm_task_0bk_place_median_rt",
        "wm_task_0bk_place_median_rt_nontarget", "wm_task_0bk_place_median_rt_target",
        "wm_task_0bk_tool_acc", "wm_task_0bk_tool_acc_nontarget",
        "wm_task_0bk_tool_acc_target", "wm_task_0bk_tool_median_rt",
        "wm_task_0bk_tool_median_rt_nontarget", "wm_task_0bk_tool_median_rt_target",
        "wm_task_2bk_acc", "wm_task_2bk_body_acc", "wm_task_2bk_body_acc_nontarget",
        "wm_task_2bk_body_acc_target", "wm_task_2bk_body_median_rt",
        "wm_task_2bk_body_median_rt_nontarget", "wm_task_2bk_body_median_rt_target",
        "wm_task_2bk_face_acc", "wm_task_2bk_face_acc_nontarget",
        "wm_task_2bk_face_acc_target", "wm_task_2bk_face_median_rt",
        "wm_task_2bk_face_median_rt_nontarget", "wm_task_2bk_face_median_rt_target",
        "wm_task_2bk_median_rt", "wm_task_2bk_place_acc",
        "wm_task_2bk_place_acc_nontarget", "wm_task_2bk_place_acc_target",
        "wm_task_2bk_place_median_rt", "wm_task_2bk_place_median_rt_nontarget",
        "wm_task_2bk_place_median_rt_target", "wm_task_2bk_tool_acc",
        "wm_task_2bk_tool_acc_nontarget", "wm_task_2bk_tool_acc_target",
        "wm_task_2bk_tool_median_rt", "wm_task_2bk_tool_median_rt_nontarget",
        "wm_task_2bk_tool_median_rt_target", "wm_task_acc", "wm_task_median_rt",
    ]
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


def load_hcp_phenotypic(focus: PhenotypicFocus = PhenotypicFocus.Reduced) -> DataFrame:
    """Get: site, dx, dx_dsm_iv, age, sex"""

    source = FreesurferStatsDataset.HCP.phenotypic_file()
    sep = "," if source.suffix == ".csv" else "\t"
    df = pd.read_csv(source, sep=sep)
    extra = focus.hcp_dict_file()
    feats = pd.read_csv(extra)
    return cleanup_HCP_phenotypic(df, extra=feats)


def reformat_HPC(df: DataFrame, keep_corr_relevant: bool = False) -> DataFrame:
    path = cast(Path, FreesurferStatsDataset.HCP.freesurfer_files())
    df = pd.read_csv(path)
    df.rename(
        columns={"Subject": "sid", "Gender": "sex", "Release": "release"}, inplace=True
    )
    df.rename(columns=str.lower, inplace=True)
    df.rename(columns=lambda s: s.replace("fs_r_", "rh-"), inplace=True)
    df.rename(columns=lambda s: s.replace("fs_l_", "lh-"), inplace=True)
    drop_reg = "curv|foldind|thckstd|numvert|_range|_min|_max|_std|intens_mean|_vox"
    if keep_corr_relevant:
        keep_reg = "sid|gender|_grayvol|_area|_thck|curv|foldind|thckstd"
    else:
        keep_reg = "sid|gender|_grayvol|_area|_thck"
    df = df.filter(regex=keep_reg).copy()
    df["sid"] = df["sid"].astype(str)
    df["sname"] = df["sid"].astype(str)
    df["parent"] = str(path.parent)
    df["fname"] = path.name
    meta_cols = ["sid", "sname", "parent", "fname"]
    cols = list(set(df.columns).difference(meta_cols))
    df = df.loc[:, meta_cols + cols].copy()
    df = df.melt(id_vars=meta_cols, var_name="metric").sort_values(by=["sid", "metric"])

    feature_classes = {
        "grayvol": "GrayVol",
        "area": "SurfArea",
        "thck": "ThickAvg",
        "thckstd": "ThickStd",
        "curvind": "CurvInd",
        "gauscurv": "GaussCurv",
        "meancurv": "MeanCurv",
        "foldind": "FoldIndex",
    }
    feat_tables = {}
    for feat in feature_classes.keys():
        idx = df["metric"].str.contains(f"{feat}$")
        feat_tables[feat] = (
            df[idx].sort_values(by=["sid", "metric"]).reset_index(drop=True)
        )

    # vols = (
    #     df[df["metric"].str.contains("grayvol")]
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # areas = (
    #     df[df["metric"].str.contains("area")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # thick = (
    #     df[df["metric"].str.contains("thck") & (~df["metric"].str.contains("thckstd"))]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # thick_sd = (
    #     df[df["metric"].str.contains("thckstd")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # curv_ind = (
    #     df[df["metric"].str.contains("curvind")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # gauss_curv = (
    #     df[df["metric"].str.contains("gauscurv")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # mean_curv = (
    #     df[df["metric"].str.contains("meancurv")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    # fold = (
    #     df[df["metric"].str.contains("foldind")]
    #     .copy()
    #     .sort_values(by=["sid", "metric"])
    #     .reset_index(drop=True)
    # )
    table: DataFrame
    for feat, table in feat_tables.items():
        table.loc[:, "StructName"] = table["metric"].str.replace(f"_{feat}", "")
        table.drop(columns="metric", inplace=True)
        table.rename(columns={"value": feature_classes[feat]}, inplace=True)

    # vols.loc[:, "StructName"] = vols["metric"].str.replace("_grayvol", "")
    # areas.loc[:, "StructName"] = areas["metric"].str.replace("_area", "")
    # thick.loc[:, "StructName"] = thick["metric"].str.replace("_thck", "")

    # vols.drop(columns="metric", inplace=True)
    # areas.drop(columns="metric", inplace=True)
    # thick.drop(columns="metric", inplace=True)

    # vols.rename(columns={"value": "GrayVol"}, inplace=True)
    # areas.rename(columns={"value": "SurfArea"}, inplace=True)
    # thick.rename(columns={"value": "ThickAvg"}, inplace=True)

    # add key columns to first table, which is the "grayvol" table in this case
    for feat, featcap in feature_classes.items():
        if feat == "grayvol":
            continue
        feat_tables["grayvol"][featcap] = feat_tables[feat][featcap]
    # feat_tables["grayvol"]["SurfArea"] = feat_tables["area"]["SurfArea"]
    # feat_tables["grayvol"]["ThickAvg"] = feat_tables["thick"]["ThickAvg"]

    df = feat_tables["grayvol"].copy()
    df["Struct"] = df["StructName"].str.replace("lh-", "")
    df["Struct"] = df["Struct"].str.replace("rh-", "")
    df["hemi"] = df["StructName"].apply(
        lambda s: "left" if s.startswith("lh") else "right"
    )
    df.loc[
        :, meta_cols + ["StructName", "Struct", "hemi", *feature_classes.values()]
    ].copy()
    return df


def load_HCP_CMC_table(keep_corr_relevant: bool = False) -> DataFrame:
    path: Path = cast(Path, FreesurferStatsDataset.HCP.freesurfer_files())
    df = pd.read_csv(path)
    df = reformat_HPC(df, keep_corr_relevant=keep_corr_relevant)
    df["pseudoVolume"] = df["ThickAvg"] * df["SurfArea"]
    df = add_bilateral_stats(df)
    df = compute_cmcs(df)
    return df


@MEMORY.cache
def load_HCP_complete(
    *,
    focus: PhenotypicFocus = PhenotypicFocus.Reduced,
    keep_corr_relevant: bool = False,
    reduce_targets: bool,
    reduce_cmc: bool,
) -> DataFrame:
    df = load_HCP_CMC_table(keep_corr_relevant=keep_corr_relevant)
    df = to_wide_subject_table(df)
    pheno = load_hcp_phenotypic(focus)
    shared = list(
        set(df.filter(regex="FS").columns).intersection(pheno.filter(regex="FS").columns)
    )
    df = pd.concat([df, pheno.drop(columns=shared)], axis=1)
    df.rename(columns=lambda col: col.replace("FS__PVOL", "CMC__pvol"), inplace=True)
    if reduce_targets:
        if focus is not PhenotypicFocus.All:
            raise NotImplementedError("Target reduction implemented only for full data")
        targs = df.filter(regex="TARGET__")
        cols = targs.columns
        targs = targs.rename(columns=lambda s: s.replace("TARGET__", ""))
        corrs = targs.corr()  # most correlations differ at most by 0.1 with spearman
        clusters = get_cluster_corrs(corrs)
        targs_reduced, all_loadings = reduce_HCP_clusters(data=targs, clusters=clusters)
        targs_reduced.rename(columns=lambda s: f"TARGET__REG__{s}", inplace=True)
        out = TABLES / "cluster_tables.md"
        lines = []
        for cluster, loadings, target in zip(
            clusters, all_loadings, targs_reduced.columns.to_list()
        ):
            # table = cluster.data.to_markdown(index=False, floatfmt="0.4f")
            label = str(target).replace("TARGET__REG__", "").replace("_", "-")
            lines.append("\n\n\\begin{table}")
            lines.append("\\centering")
            lines.append(loadings.iloc[1:].to_latex(index=True, float_format="%0.4f"))
            lines.append("\\footnotesize")
            lines.append(
                f"\\caption{{Synthetic target \\texttt{{{label}}} factor loadings.}}"
            )
            lines.append("\\normalsize")
            lines.append(f"\\label{{tab:{label}}}")
            lines.append("\\end{table}")
        tables = "\n".join(lines).replace("_", "\\_")
        out.write_text(tables)
        print(f"Wrote Appendix B tables to {out}")

        others = df.drop(columns=cols, errors="ignore")
        df = pd.concat([others, targs_reduced], axis=1)
    if reduce_cmc:
        cmcs = df.filter(regex="CMC__")
        cols = cmcs.columns
        cmcs = cmcs.rename(columns=lambda s: s.replace("CMC__", ""))
        corrs = cmcs.corr()
        clusters = get_cluster_corrs(corrs, min_cluster_size=3, epsilon=0.2)
        cmc_reduced = reduce_CMC_clusters(data=cmcs, clusters=clusters)
        others = df.drop(columns=cols)
        cmc_reduced.rename(columns=lambda s: f"CMC__{s}", inplace=True)
        df = pd.concat([others, cmc_reduced], axis=1)

    return df


if __name__ == "__main__":
    df_red = load_HCP_complete(
        focus=PhenotypicFocus.All,
        reduce_targets=True,
        reduce_cmc=False,
        keep_corr_relevant=True,
    )
    # df_full = load_HCP_complete(
    #     focus=PhenotypicFocus.All, reduce_targets=False, reduce_cmc=False
    # )
    print()
