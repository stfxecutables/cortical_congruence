from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from warnings import filterwarnings
import seaborn as sbn
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from matplotlib.patches import Patch
import os
import seaborn as sbn
import re
from joblib import Memory
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from math import ceil, floor, sqrt
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Index, Series
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing_extensions import Literal

from src.munging.hcp import (
    PhenotypicFocus,
    load_HCP_complete,
    load_HCP_CMC_table,
    to_wide_subject_table,
)

DATA = ROOT / "data/complete_tables"
MEMORY = Memory(ROOT / "__JOBLIB_CACHE__")


ABIDEI = (
    DATA
    / "ABIDE_I/ABIDE_I_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__autism.parquet"
)
ABIDEII = (
    DATA
    / "ABIDE_II/ABIDE_II_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__autism.parquet"
)
ADHD200 = (
    DATA
    / "ADHD200/ADHD200_all_FS_all_CMC_all_phenotypic_reduced_TARGET__CLS__adhd_spectrum.parquet"
)
HCP = (
    DATA / "HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
)
SINGLE_TABLES = {
    "ABIDE I": ABIDEI,
    "ABIDE II": ABIDEII,
    "ADHD200": ADHD200,
    "HCP": HCP,
}

FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
FIGURES.mkdir(exist_ok=True, parents=True)
TABLES.mkdir(exist_ok=True, parents=True)


def best_rect(m: int) -> tuple[int, int]:
    """returns dimensions (smaller, larger) of closest rectangle"""
    low = int(floor(sqrt(m)))
    high = int(ceil(sqrt(m)))
    prods = [(low, low), (low, low + 1), (high, high), (high, high + 1)]
    for prod in prods:
        if prod[0] * prod[1] >= m:
            return prod
    raise ValueError("Unreachable!")


def to_hemi(s: str) -> str:
    return (
        "left" if "lh-" in s else "right" if "rh-" in s else "both" if "bh-" in s else s
    )


def to_roi(s: str) -> str:
    return s.split("-")[-1]


CMC_FEATS = {
    "__asym__": "asym_diff",
    "__asym_abs__": "asym_diff_abs",
    "__asym_div__": "asym_div",
    "__cmc__": "cmc",
    "__SA/V__": "SA/V",
    "pseudoVolume": "pV",
}
FS_FEATS = {
    "GrayVol": "V",
    "SurfArea": "SA",
    "ThickAvg": "d",
    "FS_THCK": "THCK",
    "FS_AREA": "AREA",
    "ThickStd": "d_sd",
    "CurvInd": "CurvInd",
    "GaussCurv": "GaussCurv",
    "MeanCurv": "MeanCurv",
    "FoldIndex": "FoldIndex",
}


def to_feature_class(s: str) -> str:
    names = {**CMC_FEATS, **FS_FEATS}
    for name, fclass in names.items():
        if name in s:
            return fclass
    return "unk"


def add_feat_info(df: DataFrame) -> DataFrame:
    df = df.copy()
    if df.index.name == "feature":
        source = df.index.to_series().apply(lambda s: "FS" if "FS" in s else "CMC")
        hemi = df.index.to_series().apply(to_hemi)
        roi = df.index.to_series().apply(to_roi)
        fclass = df.index.to_series().apply(to_feature_class)
    else:
        source = df.feature.apply(lambda s: "FS" if "FS" in s else "CMC")
        hemi = df.feature.apply(to_hemi)
        roi = df.feature.apply(to_roi)
        fclass = df.feature.apply(to_feature_class)

    df.insert(1, "source", source)
    df.insert(1, "hemi", hemi)
    df.insert(1, "roi", roi)
    df.insert(1, "fclass", fclass)
    # only one is significant after multiple comparisons
    df = df[df.hemi != "none"]
    return df


def load_df(path: Path) -> DataFrame:
    return pd.read_parquet(path)


def target_hists() -> None:
    matplotlib.use("QtAgg")

    files = sorted(DATA.rglob("*.parquet"))
    targets = [
        re.search(r"TARGET__(:?(REG|CLS|MULTI))__(.*)\.parquet", str(file)).groups()[2]
        for file in files
    ]
    # dfs = [pd.read_parquet(file) for file in tqdm(files)]
    dfs = Parallel(n_jobs=-1)(delayed(load_df)(file) for file in tqdm(files))

    dfs = [pd.read_parquet(file).filter(regex="TARGET") for file in tqdm(files)]
    dfs = [df for df in dfs if df.shape[1] == 1]
    dfs = [df for df in dfs if "REG" in str(df.columns[0])]
    names = set()
    [names.update(df.columns.tolist()[0]) for df in dfs]
    dfs = [df for df in dfs if df.columns.tolist()[0] not in names]

    nrows, ncols = best_rect(len(dfs))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    ax: Axes
    for df, ax in zip(dfs, axes.flat):
        ax.hist(df, color="black", bins=50)
        ax.set_title(str(df.columns.tolist()[0]).replace("TARGET__REG__", ""))
    fig.set_size_inches(w=20, h=20)
    fig.tight_layout()
    plt.show()


def get_hemi_data(df: DataFrame) -> tuple[dict[str, DataFrame], DataFrame, DataFrame]:
    sex = df.filter(regex="DEMO(:?__FEAT)?__sex").map(
        lambda x: "M" if x in [1, "M"] else "F" if x in [0, "F"] else "NA"
    )
    age = df.filter(regex="DEMO(:?__FEAT)?__age").rename(columns=lambda s: "age")
    sex.columns = ["sex"]
    pvs = df.filter(regex="pseudoVolume").columns.tolist()
    asym = df.filter(regex="CMC__FEAT__asym__").columns.tolist()
    asym_div = df.filter(regex="CMC__FEAT__asym_div").columns.tolist()
    asym_abs = df.filter(regex="CMC__FEAT__asym_abs").columns.tolist()
    drps = pvs + asym + asym_div + asym_abs

    bh = df.filter(regex="CMC__FEAT.*__bh").drop(columns=drps, errors="ignore")
    lh = df.filter(regex="CMC__FEAT.*__lh").drop(columns=drps, errors="ignore")
    rh = df.filter(regex="CMC__FEAT.*__rh").drop(columns=drps, errors="ignore")
    rl = lh.columns.to_list() + rh.columns.to_list()

    asym = df.filter(regex="CMC__FEAT__asym__.*bh-").drop(columns=rl, errors="ignore")
    asym_div = df.filter(regex="CMC__FEAT__asym_div.*bh-").drop(
        columns=rl, errors="ignore"
    )
    asym_abs = df.filter(regex="CMC__FEAT__asym_abs.*bh-").drop(
        columns=rl, errors="ignore"
    )
    pvs = df[pvs]

    datas = {
        "Left Lateral CMC Feature": lh,
        "Right Lateral CMC Feature": rh,
        "Bilateral CMC Feature": bh,
        "Asym (signed diff) CMC Feature": asym,
        "Asym (unsigned diff) CMC Feature": asym_abs,
        "Asym (signed ratio) CMC Feature": asym_div,
        "Pseudo-volume CMC Feature": pvs,
    }
    return datas, sex, age


def hcp_boxplots() -> None:
    matplotlib.use("QtAgg")
    df = pd.read_parquet(SINGLE_TABLES["HCP"])
    datas, sex, age = get_hemi_data(df)

    ax: Axes
    fig, axes = plt.subplots(ncols=1, nrows=len(datas), sharex=False, sharey=False)
    for k, (label, data) in tqdm(enumerate(datas.items()), total=len(datas)):
        needs_label = k == len(datas) - 1
        ax = axes[k]
        # make have cols: "sex", "feat", "value"
        dfs = pd.concat([data, sex], axis=1)
        df_stats = pd.concat(
            [dfs.mean(numeric_only=True), dfs.std(numeric_only=True, ddof=1)], axis=1
        )
        df_stats.columns = ["mean", "sd"]
        ix = df_stats.sort_values(
            by=["mean", "sd"], ascending=[False, False]
        ).index.to_list()
        dfs = dfs.melt(id_vars="sex", var_name="feat")
        # dfs = dfs.loc[ix]

        n_feat = data.shape[1]
        vargs: Mapping = dict(
            x="feat",
            hue="sex",
            order=ix,
            legend=True and needs_label,
            linewidth=0.5,
            # showfliers=False,
            flier_kws=dict(s=0.5, lw=0.5),
            # kind="boxen",
            # split=True,
            # inner="point",
            # gap=0.1,
            # bw_adjust=0.5,
            # density_norm="count",
            ax=ax,
        )
        sbn.boxenplot(data=dfs, y="value", **vargs)

        shorttitle = label.replace(" CMC Feature", "")

        ticks = np.linspace(0, n_feat, num=8, endpoint=True).round(0).astype(np.int64)
        ax.set_xticks(ticks, labels=[str(t) for t in ticks], minor=False)
        ax.set_title(f"{shorttitle}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("Feature Index")
    fig.supylabel("Feature Values")
    fig.set_size_inches(h=16, w=18)
    fig.tight_layout()
    fig.subplots_adjust(
        hspace=0.3, wspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95
    )
    out = FIGURES / "HCP_boxplots.png"
    fig.savefig(out, dpi=600)
    plt.close()


def hcp_d_p_plots() -> None:
    matplotlib.use("QtAgg")
    df = pd.read_parquet(SINGLE_TABLES["HCP"])
    datas, sex, age = get_hemi_data(df)

    ax: Axes
    fig, axes = plt.subplots(ncols=1, nrows=len(datas), sharex=False, sharey=False)
    for k, (label, data) in tqdm(enumerate(datas.items()), total=len(datas)):
        needs_label = k == len(datas) - 1
        ax = axes[k]
        # make have cols: "sex", "feat", "value"
        dfs = pd.concat([data, sex], axis=1)
        dfm = dfs[dfs["sex"] == "M"].drop(columns="sex")
        dff = dfs[dfs["sex"] == "F"].drop(columns="sex")
        sd_pooled = data.std(ddof=1)
        cohen_ds = (dfm.mean() - dff.mean()) / sd_pooled
        us, ps = mannwhitneyu(x=dfm, y=dff, axis=0, nan_policy="omit")
        ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]
        # only one is significant after multiple comparisons
        sig = dfs.iloc[:, :-1].columns[ps_corrected < 0.05].to_list()

        df_stats = pd.concat(
            [dfs.mean(numeric_only=True), dfs.std(numeric_only=True, ddof=1)], axis=1
        )
        df_stats.columns = ["mean", "sd"]
        ix = df_stats.sort_values(
            by=["mean", "sd"], ascending=[False, False]
        ).index.to_list()
        dfs = dfs.melt(id_vars="sex", var_name="feat")
        # dfs = dfs.loc[ix]

        n_feat = data.shape[1]
        vargs: Mapping = dict(
            x="feat",
            hue="sex",
            order=ix,
            legend=True and needs_label,
            linewidth=0.5,
            # showfliers=False,
            flier_kws=dict(s=0.5, lw=0.5),
            # kind="boxen",
            # split=True,
            # inner="point",
            # gap=0.1,
            # bw_adjust=0.5,
            # density_norm="count",
            ax=ax,
        )
        sbn.boxenplot(data=dfs, y="value", **vargs)

        shorttitle = label.replace(" CMC Feature", "")

        ticks = np.linspace(0, n_feat, num=8, endpoint=True).round(0).astype(np.int64)
        ax.set_xticks(ticks, labels=[str(t) for t in ticks], minor=False)
        ax.set_title(f"{shorttitle} - N significant: {len(sig)}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("Feature Index")
    fig.supylabel("Feature Values")
    fig.set_size_inches(h=16, w=18)
    fig.tight_layout()
    fig.subplots_adjust(
        hspace=0.45, wspace=0.2, left=0.05, right=0.95, bottom=0.05, top=0.95
    )
    out = FIGURES / "HCP_d_p_plots.png"
    fig.savefig(out, dpi=600)
    print(f"Saved plot to {out}")
    plt.close()


def figure1() -> None:
    matplotlib.use("QtAgg")
    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        datas, sex, age = get_hemi_data(df)
        ix_male = (sex == "M").values.ravel()
        ix_female = (sex == "F").values.ravel()
        n_male, n_female = ix_male.sum(), ix_female.sum()

        ax: Axes
        fig, axes = plt.subplots(nrows=3, ncols=len(datas), sharex=False, sharey=False)
        for k, (label, data) in enumerate(datas.items()):
            needs_label = k == len(datas) - 1
            ax = axes[0][k]
            dfm = data.loc[ix_male]
            dff = data.loc[ix_female]
            n_feat = data.shape[1]

            df_m_mean = Series(name="M", data=dfm.mean())
            df_f_mean = Series(name="F", data=dff.mean())
            df_m_sd = Series(name="M", data=dfm.std(ddof=1))
            df_f_sd = Series(name="F", data=dff.std(ddof=1))

            df_means = (
                pd.concat([df_m_mean, df_f_mean], axis=1)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "sex", 0: "mean"})
            )
            df_sds = (
                pd.concat([df_m_sd, df_f_sd], axis=1)
                .stack()
                .reset_index()
                .rename(columns={"level_1": "sex", 0: "sd"})
            )
            sargs: Mapping = dict(ax=ax, alpha=0.9, s=5.0)
            sbn.scatterplot(
                x=df_m_mean,
                y=df_m_sd,
                label="M" if needs_label else None,
                color="#0a85f0",
                **sargs,
            )
            sbn.scatterplot(
                x=df_f_mean,
                y=df_f_sd,
                label="F" if needs_label else None,
                color="#da72ca",
                **sargs,
            )
            shorttitle = label.replace(" CMC Feature", "")
            ax.set_title(f"{shorttitle}\nn_feat={n_feat}")
            ax.set_xlabel("Feature Mean")
            ax.set_ylabel("Feature Standard Deviation")

            vargs: Mapping = dict(
                hue="sex",
                split=True,
                legend=True and needs_label,
                inner="point",
                gap=0.1,
                bw_adjust=0.5,
                density_norm="count",
            )
            sbn.violinplot(data=df_means, y="mean", ax=axes[1][k], **vargs)
            sbn.violinplot(data=df_sds, y="sd", ax=axes[2][k], **vargs)

        fig.suptitle(
            f"{dsname} CMC Feature Category Means and SDs by Sex (M={n_male}, F={n_female})"
        )
        fig.set_size_inches(w=13, h=7)
        fig.tight_layout()
        out = FIGURES / f"{dsname.replace(' ', '_')}_figure1.png"
        fig.savefig(out, dpi=500)
        print(f"Saved plot to {out}")
        plt.close()


def lateral_tables() -> None:
    matplotlib.use("QtAgg")
    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        datas = get_hemi_data(df)[0]

        lh = datas["Left Lateral CMC Feature"]
        rh = datas["Right Lateral CMC Feature"]
        lh = lh.rename(columns=lambda s: s.replace("__lh", "__rh"))
        # sd_pooled = pd.concat([lh, rh], axis=0).std(ddof=1)
        sd_pooled = pd.concat([lh, rh], axis=0).std(ddof=1)
        cohen_ds = (lh.mean() - rh.mean()) / sd_pooled

        lh_sd, rh_sd = lh.std(ddof=1), rh.std(ddof=1)
        lh_iqr = lh.quantile([0.25, 0.75]).T.diff(axis=1).iloc[:, 1].abs()
        rh_iqr = rh.quantile([0.25, 0.75]).T.diff(axis=1).iloc[:, 1].abs()

        sd_diff = lh_sd.mean() - rh_sd.mean()
        iqr_diff = lh_iqr.mean() - rh_iqr.mean()

        sd_sd_pooled = pd.concat([lh_sd, rh_sd], axis=0, ignore_index=True).std(ddof=1)
        iqr_sd_pooled = pd.concat([lh_iqr, rh_iqr], axis=0, ignore_index=True).std(ddof=1)

        d_sd = sd_diff / sd_sd_pooled
        d_iqr = iqr_diff / iqr_sd_pooled

        # us, ps = mannwhitneyu(x=lh, y=rh, axis=0, nan_policy="omit")
        # ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]

        # us_iqr, ps_iqr = mannwhitneyu(x=lh_iqr, y=rh_iqr, axis=0, nan_policy="omit")
        # ps_iqr = multipletests(ps_iqr, alpha=0.05, method="holm")[1]
        # us_sd, ps_sd = mannwhitneyu(x=lh_sd, y=rh_sd, axis=0, nan_policy="omit")
        # ps_sd = multipletests(ps_sd, alpha=0.05, method="holm")[1]

        ws, wps = wilcoxon(x=lh, y=rh, axis=0)
        wps_corrected = multipletests(wps, alpha=0.05, method="holm")[1]
        ws_sd, wps_sd = wilcoxon(x=lh_sd, y=rh_sd, axis=0)
        wps_sd = multipletests(wps_sd, alpha=0.05, method="holm")[1]
        ws_iqr, wps_iqr = wilcoxon(x=lh_iqr, y=rh_iqr, axis=0)
        wps_iqr = multipletests(wps_iqr, alpha=0.05, method="holm")[1]

        df_lat = (
            DataFrame(
                {
                    "d": cohen_ds,
                    "d_𝜎": d_sd,
                    "d_IQR": d_iqr,
                    # "U": us,
                    "W": ws,
                    "W_𝜎": ws_sd,
                    "W_IQR": ws_iqr,
                    # "U (p)": ps_corrected,
                    "p": wps_corrected,
                    "p_𝜎": wps_sd,
                    "p_IQR": wps_iqr,
                },
                index=cohen_ds.index,
            )
            .sort_values(by=["p", "p_𝜎", "p_IQR"], ascending=True)
            .rename(index=lambda s: s.replace("CMC__FEAT__cmc__rh-", ""))
        )
        df_lat.index = Index(name="ROI", data=df_lat.index)
        print(df_lat.round(3).to_markdown(floatfmt="0.3f"))
        print(df_lat.round(3).to_latex(float_format="%0.3f"))
        print("Table XX: Measures of Separation of Lateral CMC Features.")
        csv = TABLES / f"{dsname}_lateral_separations.csv"
        pqt = TABLES / f"{dsname}_lateral_separations.parquet"
        df_lat.to_parquet(pqt)
        df_lat.to_csv(csv, index=True)
        print(f"Saved lateral separation table to {csv}")
        print(f"Saved lateral separation table to {pqt}")


def fs_tables() -> None:
    matplotlib.use("QtAgg")
    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        dff = df.filter(regex="FS")
        rh_cols = dff.filter(regex="rh-").columns.tolist()
        lh_cols = dff.filter(regex="lh-").columns.tolist()

        lh = dff[lh_cols].rename(columns=lambda s: s.replace("lh-", "rh-"))
        rh = dff[rh_cols]
        sd_pooled = pd.concat([lh, rh], axis=0).std(ddof=1)
        cohen_ds = (lh.mean() - rh.mean()) / sd_pooled

        # us, ps = mannwhitneyu(x=lh, y=rh, axis=0, nan_policy="omit")
        # ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]

        ws, wps = wilcoxon(x=lh, y=rh, axis=0)
        wps_corrected = multipletests(wps, alpha=0.05, method="holm")[1]
        df_lat = (
            DataFrame(
                {
                    "d": cohen_ds,
                    # "U": us,
                    # "U (p)": ps_corrected,
                    "W": ws,
                    "W (p)": wps_corrected,
                },
                index=cohen_ds.index,
            )
            # .sort_values(by=["U (p)", "W (p)"], ascending=True)
            .sort_values(by=["d", "W (p)"], ascending=[False, True], key=abs)
            .rename(index=lambda s: s.replace("CMC__FEAT__cmc__rh-", ""))
        )
        df_lat.index = Index(name="ROI", data=df_lat.index)
        df_lat.rename(
            index=lambda s: s.replace("FS__FEAT__", "")
            .replace("FS__AREA_", "Area__")
            .replace("FS__THCK", "Thick__")
            .replace("rh-", ""),
            inplace=True,
        )
        print(df_lat.round(3).to_markdown(floatfmt="0.3f"))
        print(
            df_lat.round(3).to_latex(
                float_format="%0.3f",
                escape=True,
                sparsify=True,
                longtable=True,
                label="tab:lateral-fs",
                caption=(
                    "Measures of Separation of base FreeSurfer Features "
                    "(left vs.\\ right hemisphere). d = Cohen's d, W = Wilcoxon "
                    "signed rank test, W (p) = p-value for W. Note: p-values "
                    "are adjusted for multiple comparisons using the "
                    "Holm-Bonferroni stepdown method"
                ),
            )
        )
        print("Table XX: Measures of Separation of Lateral FS Features.")
        csv = TABLES / f"{dsname}_FS_lateral_separations.csv"
        pqt = TABLES / f"{dsname}_FS_lateral_separations.parquet"
        df_lat.to_parquet(pqt)
        df_lat.to_csv(csv, index=True)
        print(f"Saved FS lateral separation table to {csv}")
        print(f"Saved FS lateral separation table to {pqt}")


def tables() -> None:
    matplotlib.use("QtAgg")
    lateral_tables()
    fs_tables()

    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        datas, sex, age = get_hemi_data(df)
        df_sds = []
        df_ages_all = []

        for k, (label, data) in enumerate(datas.items()):
            dfs = pd.concat([data, sex], axis=1)
            ix_m, ix_f = dfs["sex"] == "M", dfs["sex"] == "F"
            dfm = dfs[ix_m].drop(columns="sex")
            dff = dfs[ix_f].drop(columns="sex")
            age_m, age_f = age[ix_m], age[ix_f]
            sd_pooled = data.std(ddof=1)
            cohen_ds = (dfm.mean() - dff.mean()) / sd_pooled
            us, ps = mannwhitneyu(x=dfm, y=dff, axis=0, nan_policy="omit")
            ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]

            dfm_sd, dff_sd = dfm.std(ddof=1), dff.std(ddof=1)
            dfm_iqr = dfm.quantile([0.25, 0.75]).T.diff(axis=1).iloc[:, 1].abs()
            dff_iqr = dff.quantile([0.25, 0.75]).T.diff(axis=1).iloc[:, 1].abs()

            w, wp = wilcoxon(x=dfm_sd, y=dff_sd)  # are sds different?
            w_iqr, wp_iqr = wilcoxon(x=dfm_iqr, y=dff_iqr)  # are IQRs different?
            wp = multipletests(wp, alpha=0.05, method="holm")[1]
            wp_iqr = multipletests(wp_iqr, alpha=0.05, method="holm")[1]

            sd_diff = dfm_sd.mean() - dff_sd.mean()
            iqr_diff = dfm_iqr.mean() - dff_iqr.mean()

            sd_sd_pooled = pd.concat([dfm_sd, dff_sd], axis=0, ignore_index=True).std(
                ddof=1
            )
            iqr_sd_pooled = pd.concat([dfm_iqr, dff_iqr], axis=0, ignore_index=True).std(
                ddof=1
            )
            df_ages = []
            for col in dfm.columns:
                r, p = spearmanr(dfs[col], age, nan_policy="omit")
                r_m, p_m = spearmanr(dfm[col], age_m, nan_policy="omit")
                r_f, p_f = spearmanr(dff[col], age_f, nan_policy="omit")
                shortcol = re.sub(r".*__", "", col)
                df_ages.append(
                    DataFrame(
                        {
                            "ROI": shortcol,
                            "CMC class": label.replace(" CMC Feature", ""),
                            "r": r,
                            "r_p": p,
                            "r_M": r_m,
                            "r_M_p": p_m,
                            "r_F": r_f,
                            "r_F_p": p_f,
                        },
                        index=Series(name="ROI", data=[shortcol]),
                    )
                )
            df_age = pd.concat(df_ages, axis=0)
            p_cols = df_age.filter(regex=".*_p").columns.tolist()
            # for col in p_cols:
            #     df_age[col] = multipletests(df_age[col], alpha=0.05, method="holm")[1]

            df_age["p_min"] = df_age[p_cols].min(axis=1)
            df_ages_all.append(df_age)
            has_sigs = (df_age["p_min"] < 0.05).any()
            if has_sigs:
                print(
                    df_age.sort_values(by=["p_min"], ascending=True)
                    .round(4)
                    .to_markdown(floatfmt="0.4f", index=False)
                )
            else:
                print(f"No significant age correlations for {label}")

            d_sd = sd_diff / sd_sd_pooled
            d_iqr = iqr_diff / iqr_sd_pooled

            df_sd = DataFrame(
                {
                    "CMC class": label.replace(" CMC Feature", ""),
                    "d_𝜎": d_sd,
                    "d_IQR": d_iqr,
                    "w_𝜎": w,
                    "w_IQR": w_iqr,
                    "p_𝜎": wp,
                    "p_IQR": wp_iqr,
                },
                index=[label],
            )
            df_sds.append(df_sd)
            if wp < 0.05:
                print(f"No significant separation by SD for {label}s by Sex.")
            else:
                print(f"Significant separation by SD for {label}s by Sex:")
                print(df_sd)

            df_lr = (
                DataFrame(
                    {
                        "d": cohen_ds,
                        "U": us,
                        "U (p)": ps_corrected,
                    },
                    index=cohen_ds.index,
                )
                .sort_values(by="U (p)", ascending=True)
                .rename(index=lambda s: s.replace("CMC__FEAT__cmc__rh-", ""))
            )
            df_lr.index = Index(name="ROI", data=df_lr.index)
            reg = r"CMC__FEAT__.*__bh-" if "Bilateral" in label else r"CMC__FEAT__.*__"
            df_lr.rename(index=lambda s: re.sub(reg, "", s), inplace=True)
            if (ps_corrected < 0.05).sum() < 1:
                print(f"No significant separation for {label}s by Sex.")
            else:
                print(df_lr.round(3).to_markdown(floatfmt="0.3f"))
                print(df_lr.round(3).to_latex(float_format="%0.3f"))
                print(f"Table XX: Measures of Separation of {label}s by Sex.")

            shortlabel = re.sub(
                r"[\(\) ]+", "_", label.replace(" CMC Feature", "").lower()
            )
            csv = TABLES / f"{dsname}_{shortlabel}_sex_separations.csv"
            pqt = TABLES / f"{dsname}_{shortlabel}_sex_separations.parquet"
            df_lr.to_parquet(pqt)
            df_lr.to_csv(csv, index=True)
            print(f"Saved {label}s separation table to {csv}")
            print(f"Saved {label}s separation table to {pqt}")

        df_sd = pd.concat(df_sds, axis=0)
        print(
            df_sd.sort_values(by=["p_𝜎", "p_IQR"], ascending=True).to_markdown(
                index=False, floatfmt="0.3f"
            )
        )
        print(
            df_sd.sort_values(by=["p_𝜎", "p_IQR"], ascending=True).to_latex(
                index=False, float_format="%0.3f"
            )
        )

        df_age = pd.concat(df_ages_all, axis=0, ignore_index=True)
        p_cols = df_age.filter(regex=".*_p").columns.tolist()
        n, p = df_age[p_cols].shape
        pvals = df_age[p_cols].copy().values.ravel()
        pvals = multipletests(pvals, alpha=0.05, method="holm")[1].reshape(n, p)
        df_age.loc[:, p_cols] = pvals
        df_age["p_min"] = df_age[p_cols].min(axis=1)
        print(
            df_age.sort_values(by=["p_min"], ascending=True).to_markdown(
                index=False, floatfmt="0.4f"
            )
        )
        print(
            df_age.sort_values(by=["p_min"], ascending=True).to_latex(
                index=False, float_format="%0.4f"
            )
        )
        w, p = wilcoxon(x=df_age["r_M"], y=df_age["r_F"])
        print("Do CMC-age corelations differ significantly by sex?")
        print(f"W={w}, p={p}")


def hcp_target_stats() -> None:
    matplotlib.use("QtAgg")
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )
    df = df.filter(regex="TARGET").rename(
        columns=lambda s: s.replace("TARGET__REG__", "")
    )
    df.hist()
    plt.show()
    stats = df.describe(percentiles=[0.025, 0.25, 0.75, 0.975]).T.drop(columns=["count"])

    print(stats.round(3).to_markdown(floatfmt="0.3f"))
    print(stats.round(3).to_latex(float_format="%0.3f"))


def hcp_cmc_corrs() -> None:
    # fmt: off
    import matplotlib
    matplotlib.use("QtAgg")
    CMC_FEATS = {
        "__cmc__": "CMC",
        "__asym__": "asym_diff",
        "__asym_abs__": "asym_diff_abs",
        "__asym_div__": "asym_div",
        "pseudoVolume": "pV",
    }
    # fmt: on
    def to_cmc_class(s: str) -> str:
        for cls, name in CMC_FEATS.items():
            if cls in s:
                return name
        return np.nan

    def to_fs_class(s: str) -> str:
        for cls, name in FS_FEATS.items():
            if cls in s:
                return name
        return s

    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)
    # df = load_HCP_complete(
    #     focus=PhenotypicFocus.All,  # type: ignore
    #     reduce_targets=True,  # type: ignore
    #     reduce_cmc=False,  # type: ignore
    #     keep_corr_relevant=True,  # type: ignore
    # )

    df_sav = DataFrame(
        df.filter(regex="SurfArea").values / df.filter(regex="GrayVol").values,
        index=df.index,
        columns=df.filter(regex="SurfArea")
        .columns.to_series()
        .apply(lambda s: str(s).replace("SurfArea", "SA/V")),
    )
    df = pd.concat([df, df_sav], axis=1)
    dff = add_feat_info(
        df.stack().reset_index().rename(columns={"level_1": "feature", 0: "value"})
    ).drop(columns="feature")

    print("Computing correlations")
    corrs = df.corr(method="spearman")
    corrs.index.name = "x"
    corrs.columns.name = "y"
    corrs = (
        corrs.where(np.triu(np.ones(corrs.shape), 1).astype(np.bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
        .sort_values(by="r", key=abs, ascending=False)  # top vals are largest
    )

    # who_cares = (corrs.x.str.contains("CMC") & corrs.y.str.contains("CMC")) | (
    #     corrs.x.str.contains("FS") & corrs.y.str.contains("FS")
    # )
    # corrs = corrs[~who_cares]
    # corrs = add_feat_info(corrs)
    corrs["feat_x"] = corrs["x"].apply(to_feature_class)
    corrs["feat_y"] = corrs["y"].apply(to_feature_class)
    corrs["hemi_x"] = corrs["x"].apply(to_hemi)
    corrs["hemi_y"] = corrs["y"].apply(to_hemi)
    corrs["roi_x"] = corrs["x"].apply(to_roi)
    corrs["roi_y"] = corrs["y"].apply(to_roi)
    corrs.drop(columns=["x", "y"], inplace=True)  # no more info in them now
    pd.options.display.max_rows = 500
    print(corrs.groupby(["feat_x", "feat_y"]).describe().round(3))
    print(
        corrs.groupby(["feat_x", "feat_y"])
        .describe()
        .round(3)
        .to_latex(float_format="%0.3f", sparsify=True)
    )
    print("Correlations (across subjects) Between FS and CMC Features, by Feature Types")

    print(
        corrs[
            (corrs["roi_x"] == corrs["roi_y"])
            & ((corrs.hemi_x != "both") & (corrs.hemi_y != "both"))
        ]
        .groupby(["feat_x", "feat_y"])
        .describe()
        .round(3)
    )
    print(
        corrs[
            (corrs["roi_x"] == corrs["roi_y"])
            & ((corrs.hemi_x != "both") & (corrs.hemi_y != "both"))
        ]
        .groupby(["feat_x", "feat_y"])
        .describe()
        .round(3)
        .to_latex(float_format="%0.3f", sparsify=True)
    )
    print("Correlations, restricting to same ROI (ignoring hemisphere)")

    # At this point it is clear none of the asymmetric metrics are related to much, so
    # lets just focus on CMC and pV. Also, pV and GrayVol or SurfArea correlations are
    # obvious / trivial, so lets toss those.

    corrs = corrs[corrs["CMC"].isin(["CMC", "pV"])]
    corrs = corrs[~(corrs["FS"].isin(["SurfArea", "GrayVol"]))]
    corrs = corrs[~(corrs["FS"].isin(["SurfArea", "GrayVol"]) & (corrs["CMC"] == "pV"))]

    corrs.groupby(["CMC", "FS"]).describe(percentiles=[0.025, 0.1, 0.9, 0.975])
    print("Plotting...")
    sbn.displot(corrs, kind="hist", x="r", row="CMC", col="FS")
    print(
        corrs.groupby(["CMC", "FS"]).apply(
            lambda grp: (grp["r"] > 0).mean(), include_groups=False
        )
    )
    print(
        corrs.groupby(["CMC", "FS"])
        .apply(lambda grp: (grp["r"] > 0).mean(), include_groups=False)
        .to_latex(float_format="%0.3f", sparsify=True)
    )
    print("Percent of feature correlations above 0.0:")
    plt.show()

    roi_corrs = (
        corrs.groupby(["CMC", "FS", "roi"])
        .describe(percentiles=[0.05, 0.1, 0.9, 0.95])
        .round(2)
        .droplevel(0, axis="columns")
        .drop(columns=["count", "std", "min", "mean", "max", "50%"])
        .groupby(["CMC", "FS", "roi"])
        .apply(lambda g: g.sort_values(by=["5%", "95%"]))
        .droplevel([3, 4, 5])
    )

    roi_x = corrs.x.apply(lambda s: s.split("-")[-1])
    roi_y = corrs.y.apply(lambda s: s.split("-")[-1])
    same_roi = roi_x == roi_y
    all_descs = []
    for feat in FS_FEATS:
        # != below is xor, so prevents case of identical pairings
        has_feat = corrs.x.str.contains(feat) != corrs.y.str.contains(feat)
        rel = corrs[has_feat]  # relevant correlations
        rel_diff = corrs[has_feat & ~same_roi]
        rel_same = corrs[has_feat & same_roi]

        # quick and dirty way to get name of CMC highest correlating feature
        cmc = str(rel.iloc[0, 0]).split("__")[2]
        cmc_same = str(rel_same.iloc[0, 0]).split("__")[2]
        cmc_diff = str(rel_diff.iloc[0, 0]).split("__")[2]

        # rel_desc = rel.r.abs().to_frame().describe().T
        # rel_desc_diff = rel_diff.r.abs().to_frame().describe().T
        # rel_desc_same = rel_same.r.abs().to_frame().describe().T

        rel_desc = rel.r.to_frame().describe().T
        rel_desc_diff = rel_diff.r.to_frame().describe().T
        rel_desc_same = rel_same.r.to_frame().describe().T

        cmc_col = "cmc_max_corr_feature"

        rel_desc[cmc_col] = cmc
        rel_desc_same[cmc_col] = cmc_same
        rel_desc_diff[cmc_col] = cmc_diff

        rel_desc.index = Index(name="ROIs", data=["all"])
        rel_desc_diff.index = Index(name="ROIs", data=["diff"])
        rel_desc_same.index = Index(name="ROIs", data=["same"])
        descs = pd.concat([rel_desc_same, rel_desc, rel_desc_diff], axis=0)
        descs = descs.loc[:, ["min", "mean", "max", "std", cmc_col]]
        descs[cmc_col] = descs[cmc_col].apply(lambda s: "lateral" if s == "cmc" else s)
        all_descs.append(descs)

    desc = pd.concat(all_descs, keys=FS_FEATS, axis=0)
    print(desc.round(2))
    print(desc.round(3).to_latex(index=True, sparsify=True, float_format="%0.3f"))


def hcp_cmc_fs_raw_sds() -> None:
    # from src/munging/fs_stats/tabularize.py::compute_cmcs
    subclasses = [
        "__cmc__",  # GrayVol / pseudoVolume
        "__asym__",  # cmc_l - cmc_r  # PROBLEM! no matter if bh, lh, rh
        "__asym_abs__",  # |cmc_l - cmc_r|
        "__asym_div__",  # cmc_l / cmc_r
        "pseudoVolume",
        "CurvInd",
        "FoldIndex",
        "GaussCurv",
        "GrayVol",
        "MeanCurv",
        "SurfArea",
        "ThickAvg",
        "ThickStd",
    ]
    matplotlib.use("QtAgg")
    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)

    thick_area_ratios = (
        df.filter(regex="SurfArea")
        .min()
        .rename(index=lambda s: s.replace("SurfArea", ""))
        / df.filter(regex="ThickAvg")
        .min()
        .rename(index=lambda s: s.replace("ThickAvg", ""))
    ).sort_values()
    # sds = (
    #     df.apply(
    #         lambda x: np.log(
    #             (MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).ravel() + 1)
    #         )
    #     )
    #     .std(ddof=1)
    #     .to_frame()
    #     .rename(columns={0: "sd"})
    #     .reset_index()
    # )
    info = df.std(ddof=1).to_frame().rename(columns={0: "sd"}).reset_index()
    info["mean"] = df.mean().reset_index(drop=True)
    info["min"] = df.min().reset_index(drop=True)
    info["max"] = df.max().reset_index(drop=True)
    info["coef_var"] = info["sd"] / info["mean"]
    quartiles = df.quantile(q=[0.25, 0.75])
    info["iqr"] = quartiles.diff(axis=0).iloc[1].reset_index(drop=True)
    info["med"] = df.median(axis=0).reset_index(drop=True)
    info["rob_var"] = info["iqr"] / info["med"]
    info["coef_dis"] = (quartiles.diff() / quartiles.sum()).iloc[1].reset_index(drop=True)
    info = info[
        (info["sd"] > 0) & (info["sd"] < 10_000)
    ]  # eliminate a couple crazy FoldIndex vals
    info["source"] = info["feature"].apply(
        lambda f: "FS" if "FS" in f else "CMC" if "CMC" in f else f
    )
    info["subclass"] = info["feature"].copy()
    for subclass in subclasses:
        idx = info["feature"].str.contains(subclass)
        info.loc[idx, "subclass"] = subclass.replace("__", "")

    info["hemi"] = info["feature"].apply(
        lambda s: "left"
        if "__lh" in s
        else "right"
        if "__rh" in s
        else "both"
        if "__bh" in s
        else s
    )

    info["class"] = info["feature"].apply(lambda s: str(s).split("__")[2])
    # refine "asym" class to left, right, and bilateral kinds
    idx = info["class"] == "asym"

    info["subclass"] = info["class"].copy()
    info.loc[idx, "subclass"] = info.loc[idx, "feature"].apply(
        lambda s: "left" if "lh" in s else "right" if "rh" in s else "bilateral"
    )
    pd.options.display.max_rows = 100
    print(info.groupby("subclass").describe().T.round(2))
    laterals = info[info["subclass"] == "asym"]

    # sds = sds[sds["class"].isin(["left", "right", "bilateral"])]

    logged = np.log(info["sd"] + 1)
    logged = logged[logged > 0]
    sbn.set_style("white")
    ax = sbn.histplot(
        data=info,
        # binrange=(-10)
        hue="source",
        # x=logged,
        # x="coef_var",
        x="rob_var",
        # x="coef_dis",
        # x="sd",
        kde=True,
        bins=200,
        palette={"FS": "#aaa", "CMC": "black"},
        legend=False,
    )
    # ax.lines[0].set_color("grey")
    # ax.lines[1].set_color("black")

    legend_elements = [
        Patch(facecolor="#aaa", edgecolor="#aaa", label="FS"),
        Patch(facecolor="black", edgecolor="black", label="CMC"),
    ]
    ax.legend(handles=legend_elements).set_visible(True)
    plt.show()


def hcp_sa_thick_ratios() -> None:
    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)

    sa_d_ratios = (
        (
            df.filter(regex="SurfArea")
            .min()
            .rename(index=lambda s: s.replace("SurfArea", "").replace("FS__FEAT____", ""))
            / df.filter(regex="ThickAvg")
            .min()
            .rename(index=lambda s: s.replace("ThickAvg", "").replace("FS__FEAT____", ""))
        )
        .sort_values()
        .to_frame()
        .rename(columns={0: "ratio"})
        .reset_index()
    )
    sa_d_ratios["hemi"] = sa_d_ratios["feature"].apply(to_hemi)
    sa_d_ratios["feature"] = sa_d_ratios["feature"].apply(lambda s: s.split("-")[1])
    sa_d_ratios = (
        sa_d_ratios.pivot(index="feature", columns=["hemi"], values="ratio")
        .sort_values(by=["left", "right"])
        .drop(columns="both")
    )

    v_sa_max_ratios = (
        (
            df.filter(regex="GrayVol")
            .max()
            .rename(index=lambda s: s.replace("GrayVol", "").replace("FS__FEAT____", ""))
            / df.filter(regex="SurfArea")
            .min()
            .rename(index=lambda s: s.replace("SurfArea", "").replace("FS__FEAT____", ""))
        )
        .sort_values()
        .to_frame()
        .rename(columns={0: "Vol/Area"})
        .reset_index()
    )
    v_sa_max_ratios["hemi"] = v_sa_max_ratios["feature"].apply(to_hemi)
    v_sa_max_ratios["feature"] = v_sa_max_ratios["feature"].apply(
        lambda s: s.split("-")[1]
    )
    v_sa_max_ratios = (
        v_sa_max_ratios.pivot(index="feature", columns=["hemi"], values="Vol/Area")
        .sort_values(by=["left", "right"])
        .drop(columns="both")
    )

    print(sa_d_ratios.round(1).to_markdown(tablefmt="simple"))
    print(sa_d_ratios.round(1).to_latex(index=True, float_format="%0.1f"))
    print("Ratio of ROI surface area to ROI average thickness, by hemisphere")
    # below are all above 0.975
    print("Left-right ratio correlations:")
    print(f"Spearman: {sa_d_ratios.corr(method='spearman').iloc[0, 1]}")
    print(f"Pearson: {sa_d_ratios.corr().iloc[0, 1]}")

    # table with columns: feature, min, max
    cmc_extremes = df.filter(regex="__cmc__").describe().T[["min", "max"]].reset_index()
    cmc_extremes["hemi"] = cmc_extremes["feature"].apply(to_hemi)
    cmc_extremes["feature"] = cmc_extremes["feature"].apply(lambda s: s.split("-")[1])
    cmc_extremes = cmc_extremes[cmc_extremes.hemi != "both"].pivot(
        index="feature", columns="hemi"
    )

    # these are both above 0.90, so we can basically just ignore one hemi to simplify
    c = cmc_extremes.copy()
    min_corr = c.corr(method="spearman").loc[("min", "left"), ("min", "right")]
    max_corr = c.corr(method="spearman").loc[("max", "left"), ("max", "right")]

    # below are all above 0.9
    print("Left-right CMC min/max correlations:")
    print("Spearman:")
    print(f"corr(L_min, R_min)={min_corr}")
    print(f"corr(L_max, R_max)={max_corr}")

    c = c.loc[:, [("min", "left"), ("max", "left")]].sort_values(by=(("min", "left")))
    cmc_ranges = c.diff(axis=1)["max"].sort_values(by="left")  # type: ignore
    cmc_ranges.columns = ["range"]
    sa_d_ratios.columns.name = None
    cmc_sa_ratio_corrs = (
        pd.concat([cmc_ranges, sa_d_ratios], axis=1)
        .corr(method="spearman")["range"]
        .iloc[1:]
    )

    print("Correlation (Spearman) Between CMC (standard) Range and Vol/SA Ratio")
    print(cmc_sa_ratio_corrs.round(2))

    return


def associations() -> None:
    """
    Note in this case the only categorical feature is sex, which is also really
    only useful for comparing univariate predictions (there aren't really any
    measures of assocation that work for continuous and categorical predictors
    both and which can also be compared). So we do NOT need to look at the
    categorical associations.

    Also, for the associations, it would seem F_p and pearson_p are identical
    (i.e. the p-values are the same). So for filtering on p-values and presenting
    stuff, it is really enough to just look at the correlations.

    """
    # fmt: off
    P_COLS = [
        "pearson_p", "spearman_p", "F_p"
    ]
    ASSOC_CONTS = [
        "pearson_r", "pearson_p", "spearman_r",
        "spearman_p", "F", "F_p", "mut_info",
    ]
    PRED_COLS = ["model", "mae", "msqe", "mdae", "r2", "var-exp"]
    # fmt: on
    HCP = ROOT / "cmc_results_clean/HCP"

    def to_assoc_tables(conts: list[Path]) -> DataFrame:
        dfs = []
        for path in conts:
            df = pd.read_csv(path, index_col=0)
            df.index.name = "feature"
            target = path.parent.parent.parent.parent.parent.name.replace(
                "TARGET__REG__", ""
            )
            df.insert(0, "target", target)
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
        for col in P_COLS:
            ps_corrected = multipletests(df[col], alpha=0.05, method="holm")[1]
            df[col] = ps_corrected
        df = add_feat_info(df)

        return df

    def to_pred_tables(cats: list[Path], conts: list[Path]) -> DataFrame:
        sex = []
        for path in cats:
            df = pd.read_csv(path, index_col=0)
            target = path.parent.parent.parent.parent.parent.name.replace(
                "TARGET__REG__", ""
            )
            df.insert(0, "target", target)
            sex.append(df)
        sex = pd.concat(sex, axis=0)

        tables = []
        for path in conts:
            df = pd.read_csv(path, index_col=0)
            df.index.name = "feature"
            target = path.parent.parent.parent.parent.parent.name.replace(
                "TARGET__REG__", ""
            )
            df.insert(0, "target", target)
            tables.append(df)
        tables = pd.concat(tables, axis=0)
        df = pd.concat([sex, tables], axis=0)
        df = add_feat_info(df)
        return df

    # fmt: off
    assocs_conts = sorted(HCP.rglob("**/associations/**/continuous_features.csv"))
    preds_cats   = sorted(HCP.rglob("**/predictions/**/categorical_features.csv"))
    preds_conts  = sorted(HCP.rglob("**/predictions/**/continuous_features.csv"))
    # fmt: on

    assocs = to_assoc_tables(assocs_conts)
    preds = to_pred_tables(cats=preds_cats, conts=preds_conts)
    print(assocs)
    print(preds)

    # number of CMC and FS featurea for each target (1011). Enough to just look
    # at one target to get some counts and stuff.
    df = assocs[assocs.target == assocs.target.unique()[0]]
    n_targets = len(assocs.target.unique())
    n_feat = len(df)
    n_cmc = df.index.str.contains("CMC").sum()  # 510AAA
    n_fs = df.index.str.contains("FS").sum()  # 500AAA

    sigs = assocs[
        (assocs.filter(regex="_p") < 0.05).any(axis=1)
    ]  # see docstring aboveAAA
    sig_counts = sigs.groupby("target").count().iloc[:, 0]
    sig_counts.name = "n_sig"
    """
    target         source        n
    emotion_rt     CMC         1.0
                   FS         27.0
    int_g_like     CMC        16.0
                   FS         63.0
    language_perf  CMC        32.0
                   FS        137.0
    p_matrices     CMC         2.0
                   FS         11.0
    wm_perf        CMC        18.0
                   FS        110.0
    """
    sig_counts = (
        sigs.groupby(["target", "source"])
        .describe()
        .round(2)
        .xs(("count"), axis="columns", level=1)
        .iloc[:, 0]
    )


def within_subj_corrs() -> None:
    # fmt: off
    import matplotlib
    matplotlib.use("QtAgg")
    # fmt: on

    FEATS = {**CMC_FEATS, **FS_FEATS}
    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)

    all_corrs, all_ps = [], []
    for sid in tqdm(
        df.index, total=len(df.index), desc="Computing within-subject correlations"
    ):
        dfs = df.loc[sid].to_frame().rename(columns={sid: "value"}).reset_index()
        dfs = add_feat_info(dfs)

        corrs = DataFrame(
            data=np.full([len(FEATS), len(FEATS)], np.nan),
            index=list(FEATS.values()),
            columns=list(FEATS.values()),
        )
        ps = DataFrame(
            data=np.full([len(FEATS), len(FEATS)], np.nan),
            index=list(FEATS.values()),
            columns=list(FEATS.values()),
        )

        filterwarnings("ignore", "An input array is constant")
        for i1, feat1 in enumerate(FEATS.values()):
            for i2, feat2 in enumerate(FEATS.values()):
                if i1 <= i2:
                    continue
                x = dfs[dfs["fclass"] == feat1]
                y = dfs[dfs["fclass"] == feat2]
                try:
                    r, p = spearmanr(x["value"].values, y["value"].values)
                except:  # noqa
                    r = p = np.nan
                corrs.loc[feat1, feat2] = r
                ps.loc[feat1, feat2] = p

        ps = ps.stack().reset_index().rename({"level_0": "x", "level_1": "y"})
        corrs = corrs.stack().reset_index().rename({"level_0": "x", "level_1": "y"})
        corrs["sid"] = sid
        ps["sid"] = sid
        all_corrs.append(corrs)
        all_ps.append(ps)

    corrs = pd.concat(all_corrs, axis=0).rename(
        columns={"level_0": "x", "level_1": "y", 0: "r"}
    )
    ps = pd.concat(all_ps, axis=0).rename(
        columns={"level_0": "x", "level_1": "y", 0: "p"}
    )
    corrs = pd.merge(corrs, ps, how="inner", on=["sid", "x", "y"], suffixes=(None, None))
    corrs["p"] = multipletests(ps["p"], method="holm")[1]

    # make insignificant correlations n
    cs = corrs.copy()
    cs["r"] = cs["p"].apply(lambda p: 1.0 if p <= 0.05 else np.nan) * cs["r"]
    sig_counts = (
        cs.drop(columns="sid").groupby(["x", "y"]).describe().loc[:, [("r", "count")]]
    )
    # line above produces sthg like
    """
                                  r
                              count
    CMC           FS
    asym_diff     CurvInd      10.0
                  FoldIndex    11.0
                  GaussCurv     5.0
                  MeanCurv      7.0
                  SA           16.0
                  V            10.0
                  d             6.0
                  d_sd          9.0
    asym_diff_abs CurvInd     449.0
                  FoldIndex   374.0
                  GaussCurv    37.0
                  MeanCurv     79.0
                  SA          623.0
                  V           471.0
                  d            18.0

    This is very helpful as the "count" columns tell the number of significant correlations, even
    though the rest of the values are inflated
    """
    corrs = corrs.drop(columns=["sid", "p"]).groupby(["x", "y"]).describe()
    corrs.loc[:, ("r", "count")] = sig_counts.astype(float)
    corrs = corrs.droplevel(0, axis=1).rename(columns={"count": "n_sig"})
    print()


def roi_corrs() -> None:
    matplotlib.use("QtAgg")
    FEATS = {**CMC_FEATS, **FS_FEATS}
    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)
    df = add_feat_info(df.stack().reset_index()).rename(columns={0: "value"})
    # grid = sbn.displot(
    #     data=df[df["fclass"].isin([*CMC_FEATS.values()][:-1])],
    #     x="value",
    #     col="roi",
    #     col_wrap=6,
    #     hue="fclass",
    #     kind="kde",
    #     height=2,
    #     aspect=1,
    # )
    # fig = plt.gcf()
    # sbn.move_legend(grid, loc="lower right")
    # fig.tight_layout()
    # plt.show()

    grid = sbn.catplot(
        data=df[df["fclass"].isin([*CMC_FEATS.values()][:-1])],
        x="value",
        y="fclass",
        kind="violin",
        col="roi",
        col_wrap=6,
        hue="fclass",
        height=2,
        aspect=1.5,
    )
    fig = plt.gcf()
    # sbn.move_legend(grid, loc="lower right")
    fig.tight_layout()
    plt.show()
    # cols = (
    #     df.columns.to_series()
    #     .apply(lambda s: s.split("__")[2:])
    #     .explode()
    #     .str.split("-")
    #     .explode()
    # )

    # corrs = df.corr(method="spearman")
    # corrs.index.name = "x"
    # corrs.columns.name = "y"
    # corrs = (
    #     corrs.where(np.triu(np.ones(corrs.shape), 1).astype(np.bool))
    #     .stack()
    #     .reset_index()
    #     .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    #     .sort_values(by="r", key=abs, ascending=False)  # top vals are largest
    #     .reset_index(drop=True)
    # )
    # x_corrs = add_feat_info(
    #     corrs.drop(columns="y").rename(columns={"x": "feature"})
    # ).drop(columns="feature")
    # y_corrs = add_feat_info(
    #     corrs.drop(columns="x").rename(columns={"y": "feature"})
    # ).drop(columns="feature")
    # corrs = pd.concat(
    #     [
    #         x_corrs.drop(columns=["r", "source"]).rename(columns=lambda s: f"{s}1"),
    #         y_corrs.drop(columns=["r", "source"]).rename(columns=lambda s: f"{s}2"),
    #         x_corrs["r"].to_frame(),
    #     ],
    #     axis=1,
    #     ignore_index=False,
    # )
    # corrs = corrs.loc[:, ["fclass1", "fclass2", "roi1", "roi2", "hemi1", "hemi2", "r"]]
    # corrs = corrs[corrs.roi1 == corrs.roi2].drop(columns="roi2")  # subset to same roi
    # corrs[corrs.fclass1 != corrs.fclass2].drop(
    #     columns=["roi1", "hemi1", "hemi2"]
    # ).groupby(["fclass1", "fclass2"]).describe()
    raise


@MEMORY.cache
def _load_violin_data() -> DataFrame:
    df = pd.read_parquet(SINGLE_TABLES["HCP"])
    sex, age = get_hemi_data(df)[1:]

    df = load_HCP_CMC_table(keep_corr_relevant=True)
    df = to_wide_subject_table(df)
    df = add_feat_info(df.stack().reset_index()).rename(columns={0: "value"})
    df = pd.merge(
        df,
        sex.reset_index(),
        how="inner",
        left_on="sid",
        right_on="sid",
        suffixes=(None, None),  # type: ignore
    )
    return df


def load_violin_data() -> DataFrame:
    return cast(DataFrame, _load_violin_data())


def sample_violin_plot(
    cmc_only: bool = True,
    sds: Literal["small", "large", "avg", "med", "custom", "all"] = "all",
) -> None:
    matplotlib.use("QtAgg")

    df = load_violin_data()
    custom = sds == "custom"

    feats = ["cmc"] if cmc_only else [*CMC_FEATS.values()][:-1]

    all_data = df[df["fclass"].isin(feats)]
    all_data = all_data[all_data["hemi"] != "both"]
    # rois = ["temporalpole", "insula", "paracentral"]  # sig sex differences
    small_sds = ["postcentral", "precentral", "superiorparietal"]
    avg_sds = ["inferiortemporal", "fusiform", "lingual"]  # lingual is most average sd
    large_sds = ["temporalpole", "frontalpole", "entorhinal"]
    rois = []
    if sds == "small":
        rois += small_sds
    if sds in ["avg", "med"]:
        rois += avg_sds
    elif sds == "large":
        rois += large_sds
    elif sds == "custom":
        rois = ["temporalpole", "lingual"]
    else:
        rois = large_sds + avg_sds + small_sds

    data = all_data[all_data["roi"].isin(rois)]

    if not cmc_only:
        # deal with scaling issues
        ix_invert = data["fclass"].isin(["asym_diff", "asym_diff_abs"])
        data.loc[ix_invert, "value"] = 1 - data.loc[ix_invert, "value"]
        data.loc[ix_invert, "fclass"] = data.loc[ix_invert, "fclass"].apply(
            lambda s: f"1 - {s}"
        )

    fclass = "Lateral CMC" if cmc_only else "CMC Feature"
    data.rename(columns={"fclass": fclass}, inplace=True)
    data.rename(columns={"value": "Metric Value"}, inplace=True)
    data.rename(columns={"roi": "ROI"}, inplace=True)
    data.loc[:, "hemi"] = data["hemi"].str.capitalize()

    x, y = fclass, "Metric Value"
    row, col, order = "hemi", "ROI", rois

    swap_orient = custom
    orient_args = {
        "x": y if swap_orient else x,
        "y": x if swap_orient else y,
        "row": col if swap_orient else row,
        "col": row if swap_orient else col,
        "row_order" if swap_orient else "col_order": order,
    }
    if custom:
        grid = sbn.catplot(
            data=data,
            kind="violin",
            y="Metric Value",
            x="ROI",
            col="hemi",
            hue="sex",
            palette={"M": "#bbb", "F": "#FFF"},
            inner="quart",  # only thing that works
            # inner="point",
            linewidth=1.0,
            split=True,
            bw_adjust=0.75,
            # height=2,
            # aspect=0.75,
        )
        grid.set_titles("{col_name} Hemisphere")
    else:
        grid = sbn.catplot(
            data=data,
            kind="violin",
            hue="sex",
            palette={"M": "#bbb", "F": "#FFF"},
            inner="quart",  # only thing that works
            # inner="point",
            linewidth=1.0,
            split=True,
            bw_adjust=0.75,
            **orient_args,
            # height=2,
            # aspect=0.75,
        )
    s1 = "Lateral CMC Metric" if cmc_only else "All CMC Metrics"
    s2 = {
        "small": "Small",
        "avg": "Average",
        "med": "Average",
        "large": "Large",
        "custom": "",
        "all": "Large and Small",
    }[sds]
    suptitle = f"{s1} Example Distributions"
    if custom:
        suptitle = f"{s1} Example ROI Distributions"
    else:
        suptitle = f"{suptitle}: {s2} SDs"
    grid.figure.suptitle(suptitle)

    grid.figure.set_size_inches(w=12, h=4)
    loc = {
        "small": "lower left",
        "avg": "lower left",
        "med": "lower left",
        "large": "lower left",
        "custom": (0.55, 0.65),
        "all": "lower left",
    }[sds]
    sbn.move_legend(grid, loc=loc)
    grid.figure.tight_layout()
    plt.show()


def two_roi_violin_plot() -> None:
    matplotlib.use("QtAgg")
    df = load_violin_data().drop(columns=["sid", "feature", "source"])
    df = df[df["fclass"].isin([*CMC_FEATS.values()])]
    df = df[df["hemi"] != "both"]

    nums = (
        df.groupby(["fclass", "roi", "hemi"])
        .mean(numeric_only=True)
        .groupby(["fclass", "roi"])
        .diff()
        .dropna()
        .droplevel("hemi")
    )
    var = df.groupby(["fclass", "roi", "hemi"]).var(numeric_only=True, ddof=1)
    cnts = df.groupby(["fclass", "roi", "hemi"]).count()["value"].to_frame() - 1
    sd_pooleds = (
        (cnts * var).groupby(["fclass", "roi"]).sum()
        / cnts.groupby(["fclass", "roi"]).sum()
    ).apply(np.sqrt)
    cohen_ds_lat = nums / sd_pooleds

    nums = (
        df.groupby(["fclass", "roi", "sex"])
        .mean(numeric_only=True)
        .groupby(["fclass", "roi"])
        .diff()
        .dropna()
        .droplevel("sex")
    )
    var = df.groupby(["fclass", "roi", "sex"]).var(numeric_only=True, ddof=1)
    cnts = df.groupby(["fclass", "roi", "sex"]).count()["value"].to_frame() - 1
    sd_pooleds = (
        (cnts * var).groupby(["fclass", "roi"]).sum()
        / cnts.groupby(["fclass", "roi"]).sum()
    ).apply(np.sqrt)
    cohen_ds_sex = nums / sd_pooleds

    all_data = df[(df["fclass"] == "cmc") & (df["hemi"] != "both")]
    # lingual has most "average" sd
    # paracentral has largest lateral separation, average sd
    # temporalpole has largest sex separation

    # rois = ["temporalpole", "lingual"]
    # rois = ["temporalpole", "lingual", "paracentral"]
    rois = ["temporalpole", "paracentral"]
    data = all_data[all_data["roi"].isin(rois)]

    fclass = "Lateral CMC"
    data.rename(columns={"fclass": fclass}, inplace=True)
    data.rename(columns={"value": "CMC Metric Value"}, inplace=True)
    data.rename(columns={"roi": "ROI"}, inplace=True)
    data.loc[:, "hemi"] = data["hemi"].str.capitalize()

    grid = sbn.catplot(
        data=data,
        kind="violin",
        y="CMC Metric Value",
        x="ROI",
        col="hemi",
        hue="sex",
        palette={"M": "#bbb", "F": "#FFF"},
        # inner="quart",  # only thing that works
        inner="point",  # only thing that works
        linewidth=1.0,
        split=True,
        bw_adjust=0.75,
        gap=0.15,
        # height=2,
        # aspect=0.75,
    )
    grid.set_titles("{col_name} Hemisphere")

    grid.figure.suptitle("Lateral CMC Example ROI Distributions")
    grid.figure.set_size_inches(w=6.5, h=4)
    sbn.move_legend(grid, loc=(0.575, 0.65))  # type: ignore
    grid.figure.tight_layout()
    print("Cohen's d's Left vs. Right:")
    print(cohen_ds_lat.loc["cmc"].loc[rois])
    print("Cohen's d's Male vs. Female:")
    print(cohen_ds_sex.loc["cmc"].loc[rois])

    print("Cohen's d's Left vs. Right:")
    print(cohen_ds_lat.loc["cmc"].loc[rois])
    print("Cohen's d's Male vs. Female:")
    print(cohen_ds_sex.loc["cmc"].loc[rois])

    out = ROOT / "paper/latex/figures/example_violins.png"
    grid.figure.savefig(out, dpi=600)
    print(f"Saved example distribution violin plots to {out}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # target_hists()
    # figure1()
    # hcp_boxplots()
    # hcp_d_p_plots()
    # lateral_tables()
    # tables()
    # fs_tables()
    # hcp_target_stats()

    # hcp_cmc_corrs()
    # hcp_cmc_fs_raw_sds()
    # hcp_sa_thick_ratios()
    # associations()
    # within_subj_corrs()
    # roi_corrs()
    # sample_violin_plot(cmc_only=True, sds="small")
    # sample_violin_plot(cmc_only=True, sds="large")
    # sample_violin_plot(cmc_only=True, sds="all")
    # sample_violin_plot(cmc_only=True, sds="custom")

    # sample_violin_plot(cmc_only=False, sds="small")
    # sample_violin_plot(cmc_only=False, sds="large")
    # sample_violin_plot(cmc_only=False, sds="all")

    two_roi_violin_plot()
