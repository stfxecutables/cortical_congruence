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
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import CMC_TABLE
from src.loading import load_abide_data


def load_x_y() -> tuple[DataFrame, Series]:
    df = pd.read_parquet(CMC_TABLE).dropna(axis=1, how="any")
    cmc: DataFrame = df.filter(regex="CMC")
    y = df["autism"]
    return cmc, y


# test if perfectly predicts sex, age, autism
# Failed theories: site does not predict autism


def test_cmc_accuracy(
    standardize: bool = False,
    abide: Literal["I", "II", "both"] = "I",
    target: Literal["autism", "age", "sex", "fiq", "viq", "piq"] = "autism",
) -> DataFrame:
    x = load_abide_data(standardize=standardize, abide=abide, regex=regex, target=target)

    regression = ["age", "fiq", "viq", "piq"]
    if target in regression:
        predictors = [LGBMRegressor, LinearRegression, SVR]
        scoring = "neg_mean_absolute_error"
        metric = "MAE (mean)"
    else:
        predictors = [LGBMClassifier, SVC]
        scoring = "roc_auc"
        metric = "AUROC (mean)"

    rows = []
    predictor: Type
    for predictor in predictors:
        if predictor in [SVR, SVC, LogisticRegression]:
            for C in [1.0]:
                args = dict(C=C)
                accs = cross_val_score(
                    predictor(**args),
                    x,
                    y,
                    cv=5 if target in regression else StratifiedKFold(),
                    scoring=scoring,
                    n_jobs=5,
                )
                if target in regression:
                    accs = -np.array(accs)
                folds = {f"fold{i + 1}": acc for i, acc in enumerate(accs)}
                rows.append(
                    DataFrame(
                        {
                            **{
                                "algorithm": f"{predictor.__name__}@C={C:1.0e}",
                                "abide": abide,
                                "standardized": standardize,
                                "target": target,
                                metric: np.mean(accs),
                            },
                            **folds,
                        },
                        index=[0],
                    ),
                )
        else:
            accs = cross_val_score(
                predictor(),
                x,
                y,
                cv=5 if target in regression else StratifiedKFold(),
                scoring=scoring,
                n_jobs=5,
            )
            folds = {f"fold{i + 1}": acc for i, acc in enumerate(accs)}
            rows.append(
                DataFrame(
                    {
                        **{
                            "algorithm": predictor.__name__,
                            "abide": abide,
                            "standardized": standardize,
                            "target": target,
                            metric: np.mean(accs),
                        },
                        **folds,
                    },
                    index=[0],
                ),
            )
    df = pd.concat(rows, axis=0, ignore_index=True)
    return df


def check_single_feature_accs() -> None:
    """No single feature gives high acc"""
    x, y = load_x_y()
    cmc = x.filter(regex="CMC_mesh")
    for col in tqdm(cmc.columns, total=len(cmc.columns), desc="Fitting SVCs"):
        x = cmc[col].values.reshape(-1, 1)
        accs = cross_val_score(SVC(C=1e4), x, y, cv=5, n_jobs=5)
        mean = np.mean(accs)
        if mean > 0.95:
            print(f"{col} mean acc:", mean)


if __name__ == "__main__":
    dfs = []
    for target in ["autism", "age", "sex", "fiq", "viq", "piq"]:
        for abide in ["I", "II", "both"]:
            for standardize in [False]:
                df = test_cmc_accuracy(standardize=standardize, target=target, abide=abide)  # type: ignore
                print(df)
                dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet("suspicious_results.parquet")
    print("Saved data to suspicious_results.parquet")
    cols = [
        "algorithm",
        "abide",
        "standardized",
        "target",
        "AUROC (mean)",
        "MAE (mean)",
        "fold1",
        "fold2",
        "fold3",
        "fold4",
        "fold5",
    ]
    print(df[cols].to_markdown(tablefmt="simple", floatfmt="0.3f"))

    sys.exit()

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
