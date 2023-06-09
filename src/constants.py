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
SPECIAL_MISSING = [
    "s_interm_prim-jensen",
    "g_subcallosal",
    "s_suborbital",
    "g_rectus",
    "g_and_s_cingul-ant",
    "s_orbital_med-olfact",
    "s_suborbital",
    "s_interm_prim-jensen",
    "s_parieto_occipital",
    "g_cingul-post-ventral",
]
ABIDE_I_ALL_ROIS = [
    "accumbens-area",
    "amygdala",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "caudate",
    "cerebellum-cortex",
    "cerebellum-white-matter",
    "choroid-plexus",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "g_and_s_cingul-ant",
    "g_and_s_cingul-mid-ant",
    "g_and_s_cingul-mid-post",
    "g_and_s_frontomargin",
    "g_and_s_occipital_inf",
    "g_and_s_paracentral",
    "g_and_s_subcentral",
    "g_and_s_transv_frontopol",
    "g_cingul-post-dorsal",
    "g_cingul-post-ventral",
    "g_cuneus",
    "g_front_inf-opercular",
    "g_front_inf-orbital",
    "g_front_inf-triangul",
    "g_front_middle",
    "g_front_sup",
    "g_ins_lg_and_s_cent_ins",
    "g_insular_short",
    "g_oc-temp_lat-fusifor",
    "g_oc-temp_med-lingual",
    "g_oc-temp_med-parahip",
    "g_occipital_middle",
    "g_occipital_sup",
    "g_orbital",
    "g_pariet_inf-angular",
    "g_pariet_inf-supramar",
    "g_parietal_sup",
    "g_postcentral",
    "g_precentral",
    "g_precuneus",
    "g_rectus",
    "g_subcallosal",
    "g_temp_sup-g_t_transv",
    "g_temp_sup-lateral",
    "g_temp_sup-plan_polar",
    "g_temp_sup-plan_tempo",
    "g_temporal_inf",
    "g_temporal_middle",
    "hippocampus",
    "inf-lat-vent",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lat_fis-ant-horizont",
    "lat_fis-ant-vertical",
    "lat_fis-post",
    "lateral-ventricle",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "non-wm-hypointensities",
    "pallidum",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "pole_occipital",
    "pole_temporal",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "putamen",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "s_calcarine",
    "s_central",
    "s_cingul-marginalis",
    "s_circular_insula_ant",
    "s_circular_insula_inf",
    "s_circular_insula_sup",
    "s_collat_transv_ant",
    "s_collat_transv_post",
    "s_front_inf",
    "s_front_middle",
    "s_front_sup",
    "s_interm_prim-jensen",
    "s_intrapariet_and_p_trans",
    "s_oc-temp_lat",
    "s_oc-temp_med_and_lingual",
    "s_oc_middle_and_lunatus",
    "s_oc_sup_and_transversal",
    "s_occipital_ant",
    "s_orbital-h_shaped",
    "s_orbital_lateral",
    "s_orbital_med-olfact",
    "s_parieto_occipital",
    "s_pericallosal",
    "s_postcentral",
    "s_precentral-inf-part",
    "s_precentral-sup-part",
    "s_suborbital",
    "s_subparietal",
    "s_temporal_inf",
    "s_temporal_sup",
    "s_temporal_transverse",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "thalamus-proper",
    "transversetemporal",
    "unsegmentedwhitematter",
    "ventraldc",
    "vessel",
    "wm-hypointensities",
]

ABIDE_II_ALL_ROIS = [
    "accumbens-area",
    "amygdala",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "caudate",
    "cerebellum-cortex",
    "cerebellum-white-matter",
    "choroid-plexus",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "g&s_cingul-ant",
    "g&s_cingul-mid-ant",
    "g&s_cingul-mid-post",
    "g&s_frontomargin",
    "g&s_occipital_inf",
    "g&s_paracentral",
    "g&s_subcentral",
    "g&s_transv_frontopol",
    "g_cingul-post-dorsal",
    "g_cingul-post-ventral",
    "g_cuneus",
    "g_front_inf-opercular",
    "g_front_inf-orbital",
    "g_front_inf-triangul",
    "g_front_middle",
    "g_front_sup",
    "g_ins_lg&s_cent_ins",
    "g_insular_short",
    "g_oc-temp_lat-fusifor",
    "g_oc-temp_med-lingual",
    "g_oc-temp_med-parahip",
    "g_occipital_middle",
    "g_occipital_sup",
    "g_orbital",
    "g_pariet_inf-angular",
    "g_pariet_inf-supramar",
    "g_parietal_sup",
    "g_postcentral",
    "g_precentral",
    "g_precuneus",
    "g_rectus",
    "g_subcallosal",
    "g_temp_sup-g_t_transv",
    "g_temp_sup-lateral",
    "g_temp_sup-plan_polar",
    "g_temp_sup-plan_tempo",
    "g_temporal_inf",
    "g_temporal_middle",
    "hippocampus",
    "inf-lat-vent",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lat_fis-ant-horizont",
    "lat_fis-ant-vertical",
    "lat_fis-post",
    "lateral-ventricle",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "non-wm-hypointensities",
    "pallidum",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "pole_occipital",
    "pole_temporal",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "putamen",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "s_calcarine",
    "s_central",
    "s_cingul-marginalis",
    "s_circular_insula_ant",
    "s_circular_insula_inf",
    "s_circular_insula_sup",
    "s_collat_transv_ant",
    "s_collat_transv_post",
    "s_front_inf",
    "s_front_middle",
    "s_front_sup",
    "s_interm_prim-jensen",
    "s_intrapariet&p_trans",
    "s_oc-temp_lat",
    "s_oc-temp_med&lingual",
    "s_oc_middle&lunatus",
    "s_oc_sup&transversal",
    "s_occipital_ant",
    "s_orbital-h_shaped",
    "s_orbital_lateral",
    "s_orbital_med-olfact",
    "s_parieto_occipital",
    "s_pericallosal",
    "s_postcentral",
    "s_precentral-inf-part",
    "s_precentral-sup-part",
    "s_suborbital",
    "s_subparietal",
    "s_temporal_inf",
    "s_temporal_sup",
    "s_temporal_transverse",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "thalamus-proper",
    "transversetemporal",
    "unsegmentedwhitematter",
    "ventraldc",
    "vessel",
    "wm-hypointensities",
]

ABIDE_II_EXTRA_REMOVE = [
    "ba1_exvivo",
    "ba2_exvivo",
    "ba3a_exvivo",
    "ba3b_exvivo",
    "ba44_exvivo",
    "ba45_exvivo",
    "ba4a_exvivo",
    "ba4p_exvivo",
    "ba6_exvivo",
    "entorhinal_exvivo",
    "mt_exvivo",
    "perirhinal_exvivo",
    "v1_exvivo",
    "v2_exvivo",
]
ABIDE_II_ALL_EXTRA_ROIS = [
    "accumbens-area",
    "amygdala",
    "bankssts",
    "caudalanteriorcingulate",
    "caudalanteriorcingulate-dkt",
    "caudalmiddlefrontal",
    "caudalmiddlefrontal-dkt",
    "caudate",
    "cerebellum-cortex",
    "cerebellum-white-matter",
    "choroid-plexus",
    "cuneus",
    "cuneus-dkt",
    "entorhinal",
    "entorhinal-dkt",
    "frontalpole",
    "fusiform",
    "fusiform-dkt",
    "g&s_cingul-ant",
    "g&s_cingul-mid-ant",
    "g&s_cingul-mid-post",
    "g&s_frontomargin",
    "g&s_occipital_inf",
    "g&s_paracentral",
    "g&s_subcentral",
    "g&s_transv_frontopol",
    "g_cingul-post-dorsal",
    "g_cingul-post-ventral",
    "g_cuneus",
    "g_front_inf-opercular",
    "g_front_inf-orbital",
    "g_front_inf-triangul",
    "g_front_middle",
    "g_front_sup",
    "g_ins_lg&s_cent_ins",
    "g_insular_short",
    "g_oc-temp_lat-fusifor",
    "g_oc-temp_med-lingual",
    "g_oc-temp_med-parahip",
    "g_occipital_middle",
    "g_occipital_sup",
    "g_orbital",
    "g_pariet_inf-angular",
    "g_pariet_inf-supramar",
    "g_parietal_sup",
    "g_postcentral",
    "g_precentral",
    "g_precuneus",
    "g_rectus",
    "g_subcallosal",
    "g_temp_sup-g_t_transv",
    "g_temp_sup-lateral",
    "g_temp_sup-plan_polar",
    "g_temp_sup-plan_tempo",
    "g_temporal_inf",
    "g_temporal_middle",
    "hippocampus",
    "inf-lat-vent",
    "inferiorparietal",
    "inferiorparietal-dkt",
    "inferiortemporal",
    "inferiortemporal-dkt",
    "insula",
    "insula-dkt",
    "isthmuscingulate",
    "isthmuscingulate-dkt",
    "lat_fis-ant-horizont",
    "lat_fis-ant-vertical",
    "lat_fis-post",
    "lateral-ventricle",
    "lateraloccipital",
    "lateraloccipital-dkt",
    "lateralorbitofrontal",
    "lateralorbitofrontal-dkt",
    "lingual",
    "lingual-dkt",
    "medialorbitofrontal",
    "medialorbitofrontal-dkt",
    "middletemporal",
    "middletemporal-dkt",
    "non-wm-hypointensities",
    "pallidum",
    "paracentral",
    "paracentral-dkt",
    "parahippocampal",
    "parahippocampal-dkt",
    "parsopercularis",
    "parsopercularis-dkt",
    "parsorbitalis",
    "parsorbitalis-dkt",
    "parstriangularis",
    "parstriangularis-dkt",
    "pericalcarine",
    "pericalcarine-dkt",
    "pole_occipital",
    "pole_temporal",
    "postcentral",
    "postcentral-dkt",
    "posteriorcingulate",
    "posteriorcingulate-dkt",
    "precentral",
    "precentral-dkt",
    "precuneus",
    "precuneus-dkt",
    "putamen",
    "rostralanteriorcingulate",
    "rostralanteriorcingulate-dkt",
    "rostralmiddlefrontal",
    "rostralmiddlefrontal-dkt",
    "s_calcarine",
    "s_central",
    "s_cingul-marginalis",
    "s_circular_insula_ant",
    "s_circular_insula_inf",
    "s_circular_insula_sup",
    "s_collat_transv_ant",
    "s_collat_transv_post",
    "s_front_inf",
    "s_front_middle",
    "s_front_sup",
    "s_interm_prim-jensen",
    "s_intrapariet&p_trans",
    "s_oc-temp_lat",
    "s_oc-temp_med&lingual",
    "s_oc_middle&lunatus",
    "s_oc_sup&transversal",
    "s_occipital_ant",
    "s_orbital-h_shaped",
    "s_orbital_lateral",
    "s_orbital_med-olfact",
    "s_parieto_occipital",
    "s_pericallosal",
    "s_postcentral",
    "s_precentral-inf-part",
    "s_precentral-sup-part",
    "s_suborbital",
    "s_subparietal",
    "s_temporal_inf",
    "s_temporal_sup",
    "s_temporal_transverse",
    "superiorfrontal",
    "superiorfrontal-dkt",
    "superiorparietal",
    "superiorparietal-dkt",
    "superiortemporal",
    "superiortemporal-dkt",
    "supramarginal",
    "supramarginal-dkt",
    "temporalpole",
    "thalamus-proper",
    "transversetemporal",
    "transversetemporal-dkt",
    "unsegmentedwhitematter",
    "ventraldc",
    "vessel",
    "wm-hypointensities",
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
