from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import json
import os
import re
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from io import StringIO
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

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import numpy.core.multiarray
import pandas as pd
from df_analyze.selection.embedded import EmbedSelected
from df_analyze.selection.filter import AssocResults
from df_analyze.selection.wrapper import WrapperSelected
from joblib import Parallel, delayed
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from skimage.filters import threshold_li, threshold_otsu
from tqdm import tqdm
from typing_extensions import Literal

DATA = ROOT / "cmc_results_clean"
FS_DATA = DATA / "cmc_results_fs_only_clean"
CMC_DATA = DATA / "cmc_results_cmc_only_clean"

CLS_DROPS = ["bal-acc", "npv", "ppv", "sens", "spec", "task"]
CLS_KEEPS = ["acc", "auroc", "f1"]
CLS_ADJ = ["acc+", "auroc+", "f1+"]
REG_DROPS = ["mdae", "msqe", "r2", "task"]
REG_KEEPS = ["mae", "var-exp"]
REG_ADJ = ["mae+", "var+"]

BANNER = "=" * 81
COLS = [
    "data",
    "path",
    "model",
    "feats",
    "selection",
    "acc",
    "auroc",
    "bal-acc",
    "f1",
    "npv",
    "ppv",
    "sens",
    "spec",
    #
    "mae",
    "mdae",
    "msqe",
    "r2",
    "var-exp",
    #
    "n_feat",
    "n_cmc",
    "n_fs",
    "n_sel",
    "p_sel_cmc",
    "p_sel_feat_cmc",
    "p_sel_fs",
    "p_sel_feat_fs",
    #
    "task",
    "target",
]


def to_roi(s: str) -> DataFrame:
    """Expects feature name string"""
    is_cmc = "CMC_FEAT" in s
    hemi = "lh" if "_lh" in s else "rh" if "_rh" in s else "bh"
    pv = "PseudoVolume" in s
    reg = r"(?P<source>(FS|CMC))_FEAT_(?P<measure>(pseudoVolume)|(asym_abs)|(cmc_)(SurfArea)|(GrayVol)|(NumVert)|(GausCurv)|(MeanCurv)|(FoldInd)|(ThickAvg)|(ThickStd)|(normRange)|(normMin)|(normMean)|(normMax)|(normStdDev))_(?P<hemi>[brl]h)-(?P<roi>.*)"
    res = re.search(reg, s)
    keys = ["source", "measure", "hemi", "roi"]
    info = {}
    for key in keys:
        if res is None or res.groupdict() is None:
            info[key] = np.nan
            continue

        if key not in res.groupdict():
            info[key] = np.nan
        else:
            info[key] = res.groupdict()[key]
    return DataFrame(data=[info.values()], columns=info.keys(), index=[s])


def roi_info(s: Series) -> DataFrame:
    df = pd.concat([to_roi(col) for col in s.tolist()], axis=0)
    df = df.dropna()
    return df


def get_score_info(df: DataFrame, selected: Series) -> DataFrame:
    n_feat_tot = len(df)
    n_cmc_tot = df.feat.str.contains("CMC").sum()
    n_fs_tot = df.feat.str.contains("FS").sum()

    n_keep = len(selected)
    n_cmc_keep = selected.str.contains("CMC").sum()
    n_fs_keep = selected.str.contains("FS").sum()
    p_sel_fs = n_fs_keep / n_fs_tot if n_fs_tot != 0 else 0
    p_sel_feat_fs = n_fs_keep / n_keep if n_keep != 0 else 0

    p_sel_cmc = n_cmc_keep / n_cmc_tot if n_cmc_tot != 0 else 0
    p_sel_feat_cmc = n_cmc_keep / n_keep if n_keep != 0 else 0
    return DataFrame(
        {
            "n_feat": n_feat_tot,
            "n_cmc": n_cmc_tot,
            "n_fs": n_fs_tot,
            "n_sel": n_keep,
            "p_sel_cmc": p_sel_cmc,
            "p_sel_feat_cmc": p_sel_feat_cmc,
            "p_sel_fs": p_sel_fs,
            "p_sel_feat_fs": p_sel_feat_fs,
        },
        index=[0],
    )


def read_report_tables(file: Path) -> tuple[Series, DataFrame]:
    report = file.read_text()
    selected = None
    lines = report.split("\n")
    for line in lines:
        if line.startswith("["):
            selected = Series(data=eval(line))
    if selected is None:
        raise ValueError(f"Could not parse selected features in {file}")

    table = None
    start = 0
    stop = None
    for i, line in enumerate(lines):
        if line.startswith("| "):
            start = i + 2
            i = start
            # for wrapper reports, don't read until end
            while i < len(lines):
                line = lines[i]
                if not line.startswith("|"):
                    stop = i
                    break
                i += 1
            break

    table = lines[start:stop]
    if table is None or len(table) < 1:
        raise ValueError(f"Could not parse feature scores table in {file}")
    feats, scores = [], []
    try:
        for line in table:
            feat, score = line[1:-1].split("|")
            feat = str(feat.strip())
            score = float(score.strip())
            feats.append(feat)
            scores.append(score)
    except ValueError as e:
        tab = "\n".join(table[:10] + table[-10:])
        print(f"Got error: {e} for line: `{line}` and table:\n{tab}")

    df = DataFrame(data={"feat": feats, "score": scores})
    df["feat"] = df["feat"].astype(str)
    return selected, df


def load_embed_metrics(path: Path, model: Literal["linear", "lgbm"]) -> DataFrame:
    select_dir = path.parents[1] / "selection"
    embed_dir = select_dir / "embed"
    file = embed_dir / f"{model}_embedded_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = f"embed_{model}"
    return df


def load_wrap_metrics(path: Path, model: Literal["linear", "lgbm"]) -> DataFrame:
    select_dir = path.parents[1] / "selection"
    wrap_dir = select_dir / "wrapper"
    file = wrap_dir / "wrapper_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "wrap"
    return df


def load_assoc_metrics(path: Path) -> DataFrame:
    file = path.parents[1] / "selection/filter/association_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "assoc"
    return df

    n_keep = len(feats)
    n_cmc_keep = feats.str.contains("CMC").sum()
    n_fs_keep = feats.str.contains("FS").sum()
    p_sel_feat_fs = n_fs_keep / n_keep if n_keep != 0 else 0
    p_sel_feat_cmc = n_cmc_keep / n_keep if n_keep != 0 else 0
    return DataFrame(
        {
            "n_feat": np.nan,
            "n_cmc": n_cmc_keep,
            "n_fs": n_fs_keep,
            "n_sel": len(feats),
            "p_sel_cmc": np.nan,
            "p_sel_feat_cmc": p_sel_feat_cmc,
            "p_sel_fs": np.nan,
            "p_sel_feat_fs": p_sel_feat_fs,
        },
        index=[0],
    )


def load_pred_metrics(path: Path) -> DataFrame:
    file = path.parents[1] / "selection/filter/prediction_selection_report.md"
    selected, df = read_report_tables(file)
    df = get_score_info(df, selected)
    df["selection"] = "pred"
    return df


def info_from_path(path: Path) -> DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.drop(columns=["trainset", "5-fold"])
    df = df.pivot_table(
        index="metric",
        columns=["model", "selection", "embed_selector"],
        values="holdout",
    ).T.reset_index()
    df.rename_axis(None, axis="columns", inplace=True)

    df_lgbm = load_embed_metrics(path, "lgbm")
    df_linear = load_embed_metrics(path, "linear")
    df_wrap = load_wrap_metrics(path, model="linear")
    df_assoc = load_assoc_metrics(path)
    df_pred = load_pred_metrics(path)
    df_feats = pd.concat([df_linear, df_lgbm, df_wrap, df_assoc, df_pred], axis=0)

    target = path.parents[3].stem
    is_cls = "CLS" in target
    t = "CLS" if is_cls else "REG"
    fs_only = "fs_only" in str(path)
    cmc_only = "cmc_only" in str(path)
    feats = "FS" if fs_only else "CMC" if cmc_only else "FS+CMC"

    target = target.replace(f"TARGET__{t}__", "")
    data = path.parents[4].stem
    task = "classify" if is_cls else "regress"
    df.insert(0, "data", data)
    df["feats"] = feats
    df["task"] = task
    df["target"] = target
    df["path"] = str(path.parents[1])
    df = pd.merge(df, df_feats, how="left", on="selection")
    return df


def load_summary_df() -> DataFrame:
    paths = (
        sorted(DATA.rglob("final_performances.csv"))
        + sorted(FS_DATA.rglob("final_performances.csv"))
        + sorted(CMC_DATA.rglob("final_performances.csv"))
    )
    dfs = []
    dfs = Parallel(n_jobs=-1)(
        delayed(info_from_path)(path)
        for path in tqdm(paths, desc="Loading info from disk...")
    )

    df = pd.concat(dfs, axis=0, ignore_index=True)
    cols = df.columns.tolist()
    cols.remove("task")
    cols.remove("target")
    cols = cols + ["task", "target"]
    df = df[cols]
    return df


def adjusted_reg_df(df: DataFrame) -> DataFrame:
    reg = df[df["task"] == "regress"].dropna(axis=1, how="all")
    # reg = reg.drop(columns=["msqe", "r2", "task"]).reset_index(drop=True)
    reg = reg.drop(columns=["msqe", "var-exp", "task"]).reset_index(drop=True)
    dummy_maes = (
        (
            reg[reg.model == "dummy"]
            .drop(
                columns=["selection", "embed_selector", "r2", "mdae"],
                errors="ignore",
            )
            .groupby(["data", "model", "target"])
            .min(numeric_only=True)
        )
        .reset_index()
        .drop(columns="model")
    )
    reg["dummy_mae"] = float("nan")
    # reg["dummy_var"] = float("nan")
    reg["dummy_r2"] = float("nan")
    reg["dummy_mdae"] = float("nan")
    reg["mae+"] = float("nan")
    # reg["var+"] = float("nan")
    reg["r2+"] = float("nan")
    reg["mdae+"] = float("nan")

    for i in range(len(dummy_maes)):
        data = dummy_maes.iloc[i]["data"]
        mae = dummy_maes.iloc[i]["mae"]
        targ = dummy_maes.iloc[i]["target"]
        idx = (reg.data == data) & (reg.target == targ)
        reg.loc[idx, "dummy_mae"] = mae

    dummy_vars = (
        (
            reg[reg.model == "dummy"]
            .drop(columns=["selection", "embed_selector", "mae", "mdae"], errors="ignore")
            .groupby(["data", "model", "target"])
            .max(numeric_only=True)
        )
        .reset_index()
        .drop(columns="model")
    )
    for i in range(len(dummy_vars)):
        data = dummy_vars.iloc[i]["data"]
        # score = dummy_vars.iloc[i]["var-exp"]
        score = dummy_vars.iloc[i]["r2"]
        score = max(0, score)  # handle negative MAEs
        targ = dummy_vars.iloc[i]["target"]
        idx = (reg.data == data) & (reg.target == targ)
        reg.loc[idx, "dummy_r2"] = score

    dummy_mdaes = (
        (
            reg[reg.model == "dummy"]
            .drop(
                columns=["selection", "embed_selector", "r2", "mae"],
                errors="ignore",
            )
            .groupby(["data", "model", "target"])
            .max(numeric_only=True)
        )
        .reset_index()
        .drop(columns="model")
    )

    for i in range(len(dummy_mdaes)):
        data = dummy_mdaes.iloc[i]["data"]
        score = dummy_mdaes.iloc[i]["mdae"]
        score = max(0, score)  # handle negative MAEs
        targ = dummy_mdaes.iloc[i]["target"]
        idx = (reg.data == data) & (reg.target == targ)
        reg.loc[idx, "dummy_mdae"] = score

    reg["mae+"] = reg["dummy_mae"] - reg["mae"]  # positive indicates better than dummy
    # reg["var+"] = (
    #     reg["var-exp"] - reg["dummy_var"]
    # )  # positive indicates better than dummy
    reg["r2+"] = reg["r2"] - reg["dummy_r2"]  # positive indicates better than dummy
    reg["mdae+"] = reg["dummy_mdae"] - reg["mdae"]
    # reg.drop(columns=["dummy_mae", "dummy_var", "dummy_mdae"], inplace=True)
    reg.drop(columns=["dummy_mae", "dummy_r2", "dummy_mdae"], inplace=True)
    return reg


def adjusted_cls_df(df: DataFrame) -> DataFrame:
    cls = df[df["task"] == "classify"].dropna(axis=1, how="all").drop(columns=CLS_DROPS)
    dummys = (
        (
            cls[cls.model == "dummy"]
            .drop(columns=["selection", "embed_selector"], errors="ignore")
            .groupby(["data", "model", "target"])
            .max(numeric_only=True)
        )
        .reset_index()
        .drop(columns="model")
    )
    dnames = []
    for metric in CLS_KEEPS:
        dname = f"dummy_{metric}"
        aname = f"{metric}+"
        dnames.append(dname)
        cls[dname] = float("nan")
        for i in range(len(dummys)):
            data = dummys.iloc[i]["data"]
            score = dummys.iloc[i][metric]
            targ = dummys.iloc[i]["target"]
            idx = (cls.data == data) & (cls.target == targ)
            cls.loc[idx, dname] = score
        cls[aname] = cls[metric] - cls[dname]

    cls.drop(columns=dnames, inplace=True)

    return cls


def to_table(report: str) -> Series:
    lines = report.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| feature"):
            start = i
            break
    if start is None:
        raise RuntimeError("Could not read table in report.")
    """
    Tables look like:

    | feature                                                  |      score |
    |:---------------------------------------------------------|-----------:|
    | FS_FEAT_CurvInd_bh-parahippocampal_NAN                   |  0.000e+00 |
    | FS_FEAT_CurvInd_bh-parahippocampal-dkt_NAN               |  0.000e+00 |
    | FS_FEAT_CurvInd_bh-parahippocampal-pial_NAN              |  0.000e+00 |
    | FS_FEAT_CurvInd_bh-s_interm_prim-jensen-a2009kt_NAN      | -4.486e-07 |
    | FS_FEAT_CurvInd_lh-parahippocampal_NAN                   |  0.000e+00 |
    | FS_FEAT_CurvInd_lh-parahippocampal-dkt_NAN               |  0.000e+00 |
    | FS_FEAT_CurvInd_lh-parahippocampal-pial_NAN              |  0.000e+00 |
    | FS_FEAT_CurvInd_lh-s_interm_prim-jensen-a2009kt_NAN      | -4.486e-07 |
    ...
    """
    body = lines[start + 2 :]
    feats, scores = [], []
    for i, line in enumerate(body):
        feat, score = line[2:-1].split("|")
        scores.append(float(score.strip()))
        feats.append(feat.strip())

    return Series(data=scores, index=pd.Index(feats), name="score")


def load_summary_dfs() -> tuple[DataFrame, DataFrame]:
    out = DATA / "summary.parquet"
    df = pd.read_parquet(out) if out.exists() else load_summary_df()
    # df = load_summary_df()
    if not out.exists():
        df.to_parquet(out)
    df = df.loc[:, COLS]
    print(df.shape)
    reg = adjusted_reg_df(df)  # reg: (836, 17)
    cls = adjusted_cls_df(df)
    return reg, cls


def summarize_hcp_prediction_percents() -> None:
    reg = load_summary_dfs()[0]

    hcp = reg[reg["data"] == "HCP"]
    hcp = hcp[hcp["model"] != "dummy"]
    ix_good = (hcp["mae+"] > 0) & (hcp["r2+"] > 0)

    hcp = hcp[hcp["model"] != "dummy"]
    hcp = hcp[~hcp["model"].isin(["knn", "sgd"])]
    ps = (
        hcp.groupby(["target", "feats", "model"])
        .apply(lambda g: ((g["mae+"] > 0) & (g["r2+"] > 0)).mean(), include_groups=False)
        .reset_index()
        .rename(columns={0: "exceeds_dummy"})
    )

    caption = (
        "Proportion of model runs that exceed dummy performance. "
        "feats = feature set; model = predictive model; "
        "exceeds\\_dummy = proportion of runs with performance exceeding dummy models"
    )
    print("Proportion of HCP runs exceeding dummy performance:")
    print(
        ps.sort_values(by="exceeds_dummy", ascending=False)
        .drop_duplicates()
        .to_markdown(index=False, floatfmt="0.3f")
    )
    print(
        ps.sort_values(by="exceeds_dummy", ascending=False)
        .drop_duplicates()
        .to_latex(
            index=False,
            float_format="%0.3f",
            longtable=True,
            escape=True,
            caption=caption,
            label="tab:cmc-model-p-predictive",
        )
    )
    print(caption.replace(". ", ".\n").replace("; ", ";\n"))
    # raise

    ps = (
        hcp.groupby(["target", "feats"])
        .apply(lambda g: ((g["mae+"] > 0) & (g["r2+"] > 0)).mean(), include_groups=False)
        .reset_index()
        .rename(columns={0: "exceeds_dummy"})
    )
    print("Total proportion of all models:\n")
    print(
        ps.sort_values(
            by=["target", "feats", "exceeds_dummy"], ascending=[True, True, False]
        )
        .drop_duplicates()
        .to_markdown(index=False, floatfmt="0.3f")
    )
    print(
        ps.sort_values(
            by=["target", "feats", "exceeds_dummy"], ascending=[True, True, False]
        )
        .drop_duplicates()
        .to_latex(index=False, float_format="%0.3f")
    )
    print(
        "\n: Proportion of model runs that exceed dummy performance.\n"
        "feats = feature set;\n"
        "exceeds_dummy = proportion of runs with performance exceeding dummy models;\n"
        "{#tbl:cmc_p_predictive}\n\n"
    )

    ps = (
        hcp.groupby(["target"])
        .apply(lambda g: ((g["mae+"] > 0) & (g["r2+"] > 0)).mean(), include_groups=False)
        .reset_index()
        .rename(columns={0: "exceeds_dummy"})
    )
    print("Total proportion of all models, only considering target:")
    print(
        ps.sort_values(by=["target", "exceeds_dummy"], ascending=[True, False])
        .drop_duplicates()
        .sort_values(by="exceeds_dummy", ascending=False)
        .to_markdown(index=False, floatfmt="0.3f")
    )
    print(
        ps.sort_values(by=["target", "exceeds_dummy"], ascending=[True, False])
        .drop_duplicates()
        .sort_values(by="exceeds_dummy", ascending=False)
        .to_latex(index=False, float_format="%0.3f")
    )
    print(
        "\n: Proportion of model runs where predictions exceed dummy performance.\n"
        "exceeds_dummy = proportion of runs with performance exceeding dummy models;\n"
        "{#tbl:cmc_p_target_predictive}\n\n"
    )


if __name__ == "__main__":
    summarize_hcp_prediction_percents()
    # sys.exit()
    reg, cls = load_summary_dfs()

    # mae_select_corrs = reg.corr(numeric_only=True).loc[["mae+"]].iloc[:, 2:-2]
    """
    We don't want correlations, we want the win-counts, like in the papers I
    was looking at recently. Can either threshold on performing models, or
    compare even poor models, but then see the cases where p_sel_feat_cmc > 0.5
    """

    pd.options.display.max_rows = 500
    pd.options.display.max_info_rows = 500

    # for metric in CLS_ADJ:
    ix_good = (reg["mae+"] > 0) & (reg["r2+"] > 0)
    ix_good_hcp = ix_good & (reg["data"] == "HCP")

    hcp = reg[reg["data"] == "HCP"]
    hcp = hcp[hcp["model"] != "dummy"]
    ix_good_hcp = (hcp["mae+"] > 0) & (reg["r2+"] > 0)
    print(f"Proportion of HCP runs exceeding dummy performance: {ix_good_hcp.mean()}")
    # ix_good = reg["mae+"] > 0
    r = reg.loc[ix_good]  # r: (117, 17)
    sus = r[(r.data == "ADHD200") & (r.target == "int_g_like")].sort_values(
        by="r2+",
        ascending=False,
    )
    sus = sus[sus["selection"].isin(["embed_linear", "embed_lgbm"])]
    sus_scores = []
    sus_feats = []

    for i in range(len(sus)):
        path = sus.iloc[i]["path"]
        sel = sus.iloc[i]["selection"]
        is_lgbm = sel == "embed_lgbm"
        name = "lgbm" if is_lgbm else "linear"
        report = sorted(Path(path).rglob(f"{name}_embedded_selection_report.md"))[
            0
        ].read_text()
        scores = to_table(report)

        # broken ass jsonpickle ruined things, this won't work for some reason
        # j: EmbedSelected | dict = jsonpickle.decode(
        #     sorted(Path(path).rglob(f"{name}_embed_selection_data.json"))[0].read_text(),
        #     classes=[EmbedSelected, numpy.core.multiarray.scalar],
        #     keys=True,
        #     safe=False,
        # )

        # if isinstance(j, dict):
        #     raise NotImplementedError(
        #         "TODO: just read the damn markdown tables since jsonpickle mangled the data."
        #     )
        # else:
        #     scores = DataFrame(index=j.scores.keys(), data=j.scores.values())
        if is_lgbm:
            # A typical result is:
            #
            # >>> np.unique(scores, return_counts=True)
            # (array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8., 14.]),
            #  array([4726,  924,  189,   64,   22,   12,    3,    4,    3,    1]))
            # so importances of "1" are not that important, and including all
            # importances of 2 or greater will tend to result in about 300-500
            # features, which is similar to what the top 5% would be
            scores = scores[scores >= 2].sort_values(ascending=False)
        else:
            scores = (
                scores[scores.abs() > scores.abs().quantile(0.95)]
                .abs()
                .sort_values(ascending=False)
            )
        sus_feats.append(roi_info(scores.index))
        sus_scores.append(scores)
    counts = {}
    for feats in sus_feats:
        for roi in feats.roi:
            if roi in counts:
                counts[roi] += 1
            else:
                counts[roi] = 0
    counts = DataFrame(index=counts.keys(), data=counts.values()).sort_values(by=0)
    # print(counts)
    r = r[r["data"] == "HCP"]

    bests = (
        # r.groupby(["data", "feats", "target"])
        # .apply(lambda g: g.nlargest(1, "r2+"), include_groups=False)
        r.drop(columns="path")
        .sort_values(by="r2+", ascending=False)
        # .reset_index()
        .drop(columns=["level_3", "mdae", "mdae+", "data"], errors="ignore")
    )

    print(BANNER)
    print(
        "Regression Model Performances Sorted by R2+. "
        "Only including results where performance is superior to dummy model performance."
    )
    print(
        bests.round(3)
        .drop(columns=bests.filter(regex="n_|p_").columns)
        .loc[:, ["target", "feats", "selection", "model", "r2", "mae", "mae+"]]
        .drop_duplicates()
        .to_markdown(index=False)
    )

    caption = (
        "Performance of models exceeding dummy model performance. "
        "FS = FreeSurfer features; CMC = CMC features; FS+CMC = both FS and CMC features used; "
        "wrap = forward stepwise feature selection with a linear model; "
        "assoc = feature selection by univariate association (mutual information); "
        "pred = feature selection by (linear) univariate prediction performance (accuracy); "
        "none = no feature selection (all features used in model); "
        "lgbm = LightGBM regressor; elastic = ElasticNet; "
        "r2 = coefficient of determination; mae = mean absolute error; "
        "mae+ = improvement in MAE relative to dummy model MAE;"
    )
    print(
        bests.round(3)
        .drop(columns=bests.filter(regex="n_|p_").columns)
        .loc[:, ["target", "feats", "selection", "model", "r2", "mae", "mae+"]]
        .drop_duplicates()
        .to_latex(
            index=False,
            float_format="%0.3f",
            longtable=True,
            escape=True,
            caption=caption,
            label="tab:cmc-p-model-predictive",
        )
    )
    print(caption.replace(". ", ".\n").replace("; ", ";\n"))
    raise

    bests["cmc_win"] = bests["p_sel_feat_cmc"] > bests["p_sel_feat_fs"]
    fs_cmc_bests = bests[bests["feats"] == "FS+CMC"]
    fs_cmc_bests = fs_cmc_bests.loc[~fs_cmc_bests["selection"].isin(["wrap", "none"])]

    print(BANNER)
    print("Feature selection results for best models using both CMC and FS features.")
    print(
        fs_cmc_bests.sort_values(by="p_sel_feat_cmc", ascending=False)
        .round(3)
        .loc[
            :,
            [
                "target",
                "selection",
                "model",
                "r2",
                "mae",
                "mae+",
                "p_sel_cmc",
                "p_sel_feat_cmc",
                "cmc_win",
            ],
        ]
        .to_markdown(index=False)
    )
    print(
        fs_cmc_bests.sort_values(by="p_sel_feat_cmc", ascending=False)
        .round(3)
        .loc[
            :,
            [
                "target",
                "selection",
                "model",
                "r2",
                "mae",
                "mae+",
                "p_sel_cmc",
                "p_sel_feat_cmc",
                "cmc_win",
            ],
        ]
        .to_latex(index=False, float_format="%0.3f")
    )
    print(
        "\n: Proportion of CMC features selected for best performing models;\n"
        "p_sel_cmc = proportion of *all available* CMC features selected for final model;"
        "p_sel_feat_cmc = proportion of all *selected* features that are CMC features;"
    )
    sys.exit()
    print(
        r.groupby(["data", "feats", "target"])
        .apply(lambda g: g.nlargest(1, "r2+"), include_groups=False)
        .drop(columns="path")
        .sort_values(by="r2+", ascending=False)
        .round(3)
        .dropna(axis=0, how="any")
    )
    print(BANNER)
    print("Best Regression Model Each Selection, Dataset, and Target, by R2+")
    print(
        r.groupby(["selection", "data", "feats", "target"])
        .apply(lambda g: g.nlargest(1, "r2+"), include_groups=False)
        .drop(columns="path")
        .sort_values(by=["r2+", "selection"], ascending=False)
        .round(2)
    )
    print(
        r.groupby(["selection", "data", "target"])
        .apply(lambda g: g.nlargest(1, "r2+"), include_groups=False)
        .drop(columns="path")
        .sort_values(by=["r2+", "selection"], ascending=False)
        .round(2)
        .dropna(axis=0, how="any")
    )

    ix_good = (cls["auroc+"] > 0) & (cls["f1+"] > 0) & (cls["acc+"] > 0)
    c = cls.loc[ix_good]
    c = c[c["data"] == "HCP"]
    print(BANNER)
    print("Best Classification Model Each Dataset, and Target, by AUROC+")
    print(
        c.groupby(["data", "feats", "target"])
        .apply(lambda g: g.nlargest(2, "auroc+"), include_groups=False)
        .drop(columns="path")
        .sort_values(by="auroc+", ascending=False)
        .round(2)
    )
    print(
        c.groupby(["data", "feats", "target"])
        .apply(lambda g: g.nlargest(2, "auroc+"), include_groups=False)
        .sort_values(by="auroc+", ascending=False)
        .drop(columns="path")
        .round(2)
        .dropna(axis=0, how="any")
    )
    print(BANNER)
    print("Best Classification Model Each Selection, Dataset, and Target, by AUROC+")
    print(
        c.groupby(["selection", "data", "feats", "target"])
        .apply(lambda g: g.nlargest(1, "auroc+"), include_groups=False)
        .drop(columns="path")
        .sort_values(by=["selection", "auroc+"], ascending=False)
        .round(2)
    )
    print(
        c.groupby(["selection", "data", "feats", "target"])
        .apply(lambda g: g.nlargest(1, "auroc+"), include_groups=False)
        .drop(columns="path")
        .sort_values(
            by=["selection", "data", "feats", "target", "auroc+"], ascending=False
        )
        .round(2)
        .dropna(axis=0, how="any")
    )

    # print(reg)
    # print(cls)
    ...
