from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import re
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
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing_extensions import Literal

DATA = ROOT / "data/complete_tables"

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


def get_hemi_data(df: DataFrame) -> tuple[dict[str, DataFrame], DataFrame]:
    sex = df.filter(regex="DEMO(:?__FEAT)?__sex").map(
        lambda x: "M" if x in [1, "M"] else "F" if x in [0, "F"] else "NA"
    )
    sex.columns = ["sex"]
    ix_male = (sex == "M").values.ravel()
    ix_female = (sex == "F").values.ravel()
    n_male = ix_male.sum()
    n_female = ix_female.sum()
    pvs = df.filter(regex="pseudoVolume").columns.tolist()
    asym = df.filter(regex="CMC__FEAT__asym__").columns.tolist()
    asym_div = df.filter(regex="CMC__FEAT__asym_div").columns.tolist()
    asym_abs = df.filter(regex="CMC__FEAT__asym_abs").columns.tolist()
    drps = pvs + asym + asym_div + asym_abs

    bh = df.filter(regex="CMC__FEAT.*__bh").drop(columns=drps, errors="ignore")
    lh = df.filter(regex="CMC__FEAT.*__lh").drop(columns=drps, errors="ignore")
    rh = df.filter(regex="CMC__FEAT.*__rh").drop(columns=drps, errors="ignore")

    asym = df.filter(regex="CMC__FEAT__asym__").drop(
        columns=bh.columns.tolist(), errors="ignore"
    )
    asym_div = df.filter(regex="CMC__FEAT__asym_div").drop(
        columns=bh.columns.tolist(), errors="ignore"
    )
    asym_abs = df.filter(regex="CMC__FEAT__asym_abs").drop(
        columns=bh.columns.tolist(), errors="ignore"
    )
    pvs = df[pvs]

    datas = {
        "Left Lateral CMC Feature": lh,
        "Right Lateral CMC Feature": rh,
        "Bilateral CMC Feature": bh,
        "Asym (signed) CMC Feature": asym,
        "Asym (unsigned) CMC Feature": asym_abs,
        "Asym (ratio) CMC Feature": asym_div,
        "Pseudo-volume CMC Feature": pvs,
    }
    return datas, sex


def hcp_boxplots() -> None:
    matplotlib.use("QtAgg")
    df = pd.read_parquet(SINGLE_TABLES["HCP"])
    datas, sex = get_hemi_data(df)

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
    plt.show()


def hcp_d_p_plots() -> None:
    matplotlib.use("QtAgg")
    df = pd.read_parquet(SINGLE_TABLES["HCP"])
    datas, sex = get_hemi_data(df)
    ix_male = (sex == "M").values.ravel()
    ix_female = (sex == "F").values.ravel()
    n_male = ix_male.sum()
    n_female = ix_female.sum()

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
    plt.show()


def figure1() -> None:
    matplotlib.use("QtAgg")
    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        datas, sex = get_hemi_data(df)
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


def tables() -> None:
    matplotlib.use("QtAgg")
    for dsname, path in SINGLE_TABLES.items():
        if dsname != "HCP":
            continue
        df = pd.read_parquet(path)
        datas, sex = get_hemi_data(df)
        ix_male = (sex == "M").values.ravel()
        ix_female = (sex == "F").values.ravel()
        n_male, n_female = ix_male.sum(), ix_female.sum()

        lh = datas["Left Lateral CMC Feature"]
        rh = datas["Right Lateral CMC Feature"]
        lh = lh.rename(columns=lambda s: s.replace("__lh", "__rh"))
        sd_pooled = pd.concat([lh, rh], axis=0).std(ddof=1)
        cohen_ds = (lh.mean() - rh.mean()) / sd_pooled

        us, ps = mannwhitneyu(x=lh, y=rh, axis=0, nan_policy="omit")
        ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]

        ws, wps = wilcoxon(x=lh, y=rh, axis=0)
        wps_corrected = multipletests(wps, alpha=0.05, method="holm")[1]
        df_lat = (
            DataFrame(
                {
                    "d": cohen_ds,
                    "U": us,
                    "U (p)": ps_corrected,
                    "W": ws,
                    "W (p)": wps_corrected,
                },
                index=cohen_ds.index,
            )
            .sort_values(by=["U (p)", "W (p)"], ascending=True)
            .rename(index=lambda s: s.replace("CMC__FEAT__cmc__rh-", ""))
        )
        df_lat.index = Index(name="ROI", data=df_lat.index)
        print(df_lat.round(3).to_markdown(tablefmt="simple", floatfmt="0.3f"))
        print("Table XX: Measures of Separation of Lateral CMC Features.")
        csv = TABLES / f"{dsname}_lateral_separations.csv"
        pqt = TABLES / f"{dsname}_lateral_separations.parquet"
        df_lat.to_parquet(pqt)
        df_lat.to_csv(csv, index=True)
        print(f"Saved lateral separation table to {csv}")
        print(f"Saved lateral separation table to {pqt}")

        for k, (label, data) in enumerate(datas.items()):
            dfs = pd.concat([data, sex], axis=1)
            dfm = dfs[dfs["sex"] == "M"].drop(columns="sex")
            dff = dfs[dfs["sex"] == "F"].drop(columns="sex")
            sd_pooled = data.std(ddof=1)
            cohen_ds = (dfm.mean() - dff.mean()) / sd_pooled
            us, ps = mannwhitneyu(x=dfm, y=dff, axis=0, nan_policy="omit")
            ps_corrected = multipletests(ps, alpha=0.05, method="holm")[1]

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
                print(df_lr.round(3).to_markdown(tablefmt="simple", floatfmt="0.3f"))
                print(f"Table XX: Measures of Separation of {label}s by Sex.")

            shortlabel = re.sub(
                r"[\(\) ]+", "_", label.replace(" CMC Feature", "").lower()
            )
            csv = TABLES / f"{dsname}_{shortlabel}_sex_separations.csv"
            pqt = TABLES / f"{dsname}_{shortlabel}_sex_separations.parquet"
            df_lat.to_parquet(pqt)
            df_lat.to_csv(csv, index=True)
            print(f"Saved {label}s separation table to {csv}")
            print(f"Saved {label}s separation table to {pqt}")


if __name__ == "__main__":
    # target_hists()
    # figure1()
    # hcp_boxplots()
    # hcp_d_p_plots()
    tables()
