from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from lightgbm import LGBMClassifier as LGB
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.loading import load_abide_data

matplotlib.use("QtAgg")

"""
Notes
-----

We wish to define "abnormal" CMC values for each ROI. The most reasonable way
to do this is based on the distribution of values for autistic and control
subjects, essentially, to perform a logistic regression or linear SVC to get
the support vectors / decision boundary for each feature in the training set,
and then
"""

SINGLE_FEATURE_SEPS = ROOT / "single_feature_separations.parquet"


def plot_distributions(regex: str = "CMC_mesh") -> None:
    x, y = load_abide_data(abide="both", regex=regex, standardize=False, target="autism")
    x = x.rename(columns=lambda s: s.replace(f"__{regex}", ""))
    df = pd.concat([x, y], axis=1, ignore_index=False)
    print(df)

    label = lambda s: s if k == 0 else None
    n_figs = 48
    for i in range(0, len(x.columns), n_figs):
        fig, axes = plt.subplots(nrows=8, ncols=6, sharex=False, sharey=False)
        for k, ax in enumerate(axes.flat):
            if i + k >= len(x.columns):
                continue
            autism = x.iloc[:, i + k][y == 1]
            control = x.iloc[:, i + k][y == 0]
            ax.hist(x=control, color="black", bins=50, label=label("autism"), alpha=0.5)
            ax.hist(x=autism, color="orange", bins=50, label=label("control"), alpha=0.5)
            ax.set_title(x.iloc[:, i + k].name)
        fig.suptitle(regex)
        fig.legend().set_visible(True)
        fig.set_size_inches(w=18, h=12)
        fig.tight_layout()
        plt.show(block=False)
    plt.show()


def _column_predict(x_y_column: tuple[DataFrame, Series, Series]) -> DataFrame:
    x, y, col = x_y_column
    autism = x[col][y == 1]
    control = x[col][y == 0]
    diff = np.mean(autism) - np.mean(control)
    cohens_d = diff / np.sqrt((np.std(autism, ddof=1) ** 2 + np.std(control) ** 2) / 2)
    x_r = x[col].values.reshape(-1, 1)
    auroc_svc = roc_auc_score(y, SVC().fit(x_r, y).predict(x_r))
    auroc_lr = roc_auc_score(y, LR().fit(x_r, y).predict(x_r))
    return DataFrame(
        {
            "diff": diff,
            "cohen_d": cohens_d,
            "AUROC (SVC)": auroc_svc,
            "AUROC (LR)": auroc_lr,
        },
        index=[col],
    )


def compute_single_feature_aurocs(regex: str = "CMC") -> None:
    x, y = load_abide_data(abide="both", regex=regex, standardize=False, target="autism")
    # x = x.rename(columns=lambda s: s.replace(f"__{regex}", ""))
    df = pd.concat([x, y], axis=1, ignore_index=False)
    x_y_cols = [(x, y, col) for col in x.columns]
    dfs = process_map(_column_predict, x_y_cols, desc="Computing separations")
    df = pd.concat(dfs, axis=0, ignore_index=False)
    out = SINGLE_FEATURE_SEPS
    df.to_parquet(out)
    print(f"Saved separations to {out}")


def compute_n_feature_aurocs(
    df_x_y_n: tuple[DataFrame, DataFrame, Series, int]
) -> DataFrame:
    df, x, y, n = df_x_y_n
    metrics = df.iloc[:n].iloc[:, 0].index.to_list()
    x_sub = x.loc[:, metrics]
    auroc_lrs, auroc_svcs, auroc_lgbs = [], [], []
    for idx_tr, idx_t in StratifiedKFold().split(x_sub, y):
        x_tr, x_t = x_sub.loc[idx_tr], x_sub.loc[idx_t]
        y_tr, y_t = y.loc[idx_tr], y.loc[idx_t]
        y_pred_lr = LR(max_iter=1000).fit(x_tr, y_tr).predict(x_t)
        y_pred_svc = SVC().fit(x_tr, y_tr).predict(x_t)
        y_pred_lgb = LGB().fit(x_tr, y_tr).predict(x_t)
        auroc_lrs.append(roc_auc_score(y_t, y_pred_lr))
        auroc_svcs.append(roc_auc_score(y_t, y_pred_svc))
        auroc_lgbs.append(roc_auc_score(y_t, y_pred_lgb))
    return DataFrame(
        {
            "n": n,
            "LR": np.mean(auroc_lrs),
            "SVC": np.mean(auroc_svcs),
            "LGB": np.mean(auroc_lgbs),
            "LR_p2.5": np.percentile(auroc_lrs, 2.5),
            "LR_p97.5": np.percentile(auroc_lrs, 97.5),
            "SVC_p2.5": np.percentile(auroc_svcs, 2.5),
            "SVC_p97.5": np.percentile(auroc_svcs, 97.5),
            "LGB_p2.5": np.percentile(auroc_lgbs, 2.5),
            "LGB_p97.5": np.percentile(auroc_lgbs, 97.5),
        },
        index=[n],
    )


if __name__ == "__main__":
    # plot_distributions()
    # compute_single_feature_aurocs(regex="CMC")
    x, y = load_abide_data(abide="both", regex="CMC", standardize=False, target="autism")
    df = pd.read_parquet(SINGLE_FEATURE_SEPS)
    df.sort_values(by="AUROC (SVC)", inplace=True)
    args = [(df, x, y, n) for n in range(10, 200, 10)]
    rows = process_map(
        compute_n_feature_aurocs, args, desc="Fitting Cheaty Feature-Selected Models"
    )
    df = pd.concat(rows, axis=0, ignore_index=False)
    out = ROOT / "cheaty_feature_selected.parquet"
    df.to_parquet(out)
    print(f"Saved cheaty feature selection results to {out}")
