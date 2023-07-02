from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import re
import sys
from argparse import Namespace
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor as LGB
from pandas import DataFrame, Series
from pandas.errors import ParserError
from sklearn.cluster._hdbscan.hdbscan import HDBSCAN
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import (
    ABIDE_II_ENCODING,
    ALL_STATSFILES,
    CACHED_RESULTS,
    DATA,
    HCP_FEATURE_INFO,
    MEMORY,
    TABLES,
)
from src.enumerables import FreesurferStatsDataset, PhenotypicFocus
from src.munging.fs_stats import load_HCP_complete


def print_correlations() -> None:
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=True
    )
    dfr = df.filter(regex="CMC|TARGET|DEMO")
    corrs = (
        dfr.corr()
        .stack()
        .reset_index()
        .rename(columns={"level_0": "x", "level_1": "y", 0: "r"})
    )
    corrs["abs"] = corrs["r"].abs()
    corrs = corrs.sort_values(by="abs", ascending=False).drop(columns="abs")
    corrs = corrs[corrs["r"] < 1.0]
    cmc_corrs = corrs[
        (corrs.x.str.contains("CMC") & corrs.y.str.contains("TARGET"))
        | (corrs.x.str.contains("TARGET") & corrs.y.str.contains("CMC"))
    ]
    with pd.option_context("display.max_rows", 500):
        print(cmc_corrs)


def compute_regression_metrics(args: Namespace) -> DataFrame:
    dfr = args.data
    model = args.model
    modelname = args.modelname
    target = args.target
    x = dfr.filter(regex="CMC").to_numpy()
    y = dfr[target]
    results = cross_validate(
        model,
        x,
        y,
        cv=5,
        scoring=["neg_mean_absolute_error", "r2", "explained_variance"],
    )
    df = DataFrame(results).mean().to_frame()
    df = df.T.copy()
    df.rename(
        columns={
            "test_neg_mean_absolute_error": "MAE",
            "test_r2": "R2",
            "test_explained_variance": "Exp.Var",
        },
        inplace=True,
    )
    df.loc[:, "MAE"] = df["MAE"].abs()
    df.loc[:, "R2"] = df["R2"].abs()
    df.loc[:, "Exp.Var"] = df["Exp.Var"].abs()
    df["target"] = str(target).replace("TARGET__", "")
    df["model"] = modelname
    return df


def get_reg_args(reduce_cmc: bool, reduce_targets: bool = True) -> list[Namespace]:
    n_jobs = 1
    models = {
        "LR": lambda: LR(n_jobs=n_jobs),
        "MLP": lambda: MLP(
            hidden_layer_sizes=[32, 64, 128, 256],
            activation="relu",
            solver="adam",
            alpha=1e-4,
            shuffle=True,
            learning_rate_init=3e-4,
            max_iter=500,
        ),
        "LGB": lambda: LGB(n_jobs=n_jobs),
        "KNN-1": lambda: KNN(n_neighbors=1, n_jobs=n_jobs),
        "KNN-3": lambda: KNN(n_neighbors=3, n_jobs=n_jobs),
        "KNN-9": lambda: KNN(n_neighbors=9, n_jobs=n_jobs),
        "Dummy": lambda: Dummy(strategy="mean"),
    }
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    dfr = df.filter(regex="CMC|TARGET|DEMO")
    args = []
    for target in dfr.filter(regex="TARGET").columns:
        for model in models:
            args.append(
                Namespace(
                    **dict(
                        data=dfr, model=models[model](), modelname=model, target=target
                    )
                )
            )
    return args


if __name__ == "__main__":
    # pd.options.display.max_rows = 500
    # load_HCP_complete.clear()
    reduce_cmc = False
    args = get_reg_args(reduce_cmc=reduce_cmc)
    r = "reduced" if reduce_cmc else "raw"
    out = TABLES / f"quick_regression_cluster_targets_cmc_{r}.parquet"
    if out.exists():
        df = pd.read_parquet(out)
    else:
        all_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(compute_regression_metrics)(arg) for arg in args
        )
        df = pd.concat(all_results, axis=0).reset_index(drop=True)
        print(df.to_markdown(tablefmt="simple", floatfmt="0.4f"))
        df.to_parquet(out)
        print(f"Saved results to {out}")

    dummy_maes = (
        df.groupby("target")
        .apply(lambda grp: grp[grp.model == "Dummy"]["MAE"])
        .droplevel(1)
    )
    base_mae = df.target.apply(lambda target: dummy_maes[target])
    df.insert(2, "sMAE", df["MAE"] / base_mae)
    df = df.applymap(lambda x: np.nan if (isinstance(x, float) and x > 10) else x)
    print(df.to_markdown(tablefmt="simple", index=False, floatfmt="0.4f"))

    # print_correlations()
