from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing_extensions import Literal

TABLE = ROOT / "abide_cmc_combined.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(TABLE)
    # y = df.autism.copy()
    # x = df.filter(regex="CMC").copy()
    # x = x.loc[:, ~(x.isna().sum() > 200)]
    # acc = cross_val_score(LGBMClassifier(), x, y)
    # print("LGBM accuracy:", acc)

    df2 = df.dropna(axis=1, how="any")
    cmc: DataFrame = df2.filter(regex="CMC")
    for metric in [
        "CMC_mesh",
        "CMC_vox",
        "CMC_asym_mesh",
        "CMC_asym_vox",
        ".*mesh",
        ".*vox",
        # "CMC.*",
    ]:
        x = StandardScaler().fit_transform(cmc.filter(regex=metric))
        y = df2.autism
        # for C, n_iter in zip(
        #     [1e3, 1e4, 1e5, 1e6],
        #     [5000, 7500, 7500, 7500],
        # ):
        for C in [1e3, 1e4, 1e5, 1e6]:
            # acc = cross_val_score(
            #     LogisticRegression(max_iter=n_iter, C=C), x, y, n_jobs=5
            # )
            # pre = f"LR mean accuracy @ C={C:1.1e} using {metric}:"
            # print(
            #     f"{pre:<60}",
            #     f"{np.mean(acc).round(3)}",
            #     f" - {np.round(acc, 3)}",
            # )
            acc = cross_val_score(SVC(C=C), x, y, n_jobs=5)  # type: ignore
            pre = f"SVM mean accuracy @ C={C:1.1e} using {metric}:"
            print(
                f"{pre:<60}",
                f"{np.mean(acc).round(3)}",
                f" - {np.round(acc, 3)}",
            )

            # acc = cross_val_score(LGBMClassifier(), x, y, n_jobs=5)  # type: ignore
            # pre = f"LightGBM mean accuracy @ C={C:1.1e} using {metric}:"
            # print(
            #     f"{pre:<60}",
            #     f"{np.mean(acc).round(3)}",
            #     f" - {np.round(acc, 3)}",
            # )

    # kf = KFold(5)
    # accs, aucs = [], []
    # for idx_train, idx_test in kf.split(x, y):
    #     x_train = x.iloc[idx_train]
    #     x_test = x.iloc[idx_test]
    #     y_train = y[idx_train]
    #     y_test = y[idx_test]
    #     lgb = LGBMClassifier()
    #     lgb.fit(x_train, y_train)
    #     y_pred = lgb.predict(x_test)
    #     acc = np.mean(y_pred == y_test)
    #     auc = roc_auc_score(y_test, y_pred)
    #     accs.append(acc)
    #     aucs.append(auc)
    #     print("acc:", acc)
    #     print("AUC:", auc)
    # print(f"Mean accuracy: {np.mean(accs)}")
    # print(f"Mean AUC:      {np.mean(aucs)}")
