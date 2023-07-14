from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from argparse import Namespace
from math import ceil
from pathlib import Path
from random import shuffle
from typing import Any, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_validate, train_test_split
from tqdm import tqdm

from src.constants import PLOTS, TABLES
from src.enumerables import FreesurferStatsDataset, RegressionMetric, RegressionModel
from src.feature_selection.stepwise import StepwiseSelect
from src.munging.hcp import PhenotypicFocus

matplotlib.use("QtAgg")


def get_stepup_args(
    feature_regex: str = "CMC",
    reduce_cmc: bool = False,
    reduce_targets: bool = True,
    random_feats: bool = False,
    n_random: int = 250,
    perc_max: float = 0.25,
) -> list[Namespace]:
    n_jobs = 1
    models = {
        "LR": lambda: LR(n_jobs=n_jobs),
        # "MLP": lambda: MLP(
        #     hidden_layer_sizes=[32, 64, 128, 256],
        #     activation="relu",
        #     solver="adam",
        #     alpha=1e-4,
        #     shuffle=True,
        #     learning_rate_init=3e-4,
        #     max_iter=500,
        # ),
        # "LGB": lambda: LGB(n_jobs=n_jobs),
        # "KNN-1": lambda: KNN(n_neighbors=1, n_jobs=n_jobs),
        # "KNN-3": lambda: KNN(n_neighbors=3, n_jobs=n_jobs),
        # "KNN-9": lambda: KNN(n_neighbors=9, n_jobs=n_jobs),
        "Dummy": lambda: Dummy(strategy="mean"),
    }
    df = FreesurferStatsDataset.HCP.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=True, reduce_cmc=False
    )
    dfr = df.copy()
    args = []
    for target in dfr.filter(regex="TARGET").columns:
        for model in models:
            if not random_feats:
                args.append(
                    Namespace(
                        **dict(
                            data=dfr,
                            model=models[model](),
                            modelname=model,
                            target=target,
                            feature_regex=feature_regex,
                            random_feats=False,
                            perc_max=perc_max,
                        )
                    )
                )
            else:
                for _ in range(n_random):
                    args.append(
                        Namespace(
                            **dict(
                                data=dfr,
                                model=models[model](),
                                modelname=model,
                                target=target,
                                feature_regex=feature_regex,
                                random_feats=True,
                                perc_max=perc_max,
                            )
                        )
                    )

    shuffle(args)
    return args


def stepup_feature_select(
    regex: str,
    model: RegressionModel,
    scoring: RegressionMetric = RegressionMetric.ExplainedVariance,
    max_n_features: int = 100,
    inner_progress: bool = False,
    holdout: float | None = None,
    params: Mapping | None = None,
) -> DataFrame:
    h = f"_holdout={holdout}" if holdout is not None else ""
    fname = (
        f"stepup_{scoring.value}_selected"
        f"_{model.value}_{regex}_n={max_n_features}{h}.parquet"
    )
    scores_out = TABLES / fname
    if scores_out.exists():
        return pd.read_parquet(scores_out)

    if params is None:
        params = dict()
    reduce_cmc = False
    reduce_targets = True
    df = FreesurferStatsDataset.HCP.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    if holdout is not None:
        df, df_test = train_test_split(df, test_size=holdout)
    else:
        df_test = df

    all_scores = []
    count = 0
    # pbar_outer = tqdm(["CMC", "FS", "FS|CMC"], leave=True)
    pbar_outer = tqdm([regex], leave=True)
    # pbar_outer = tqdm(["FS", "FS|CMC"], leave=True)
    for feature_regex in pbar_outer:
        pbar_outer.set_description(feature_regex)
        features = df.filter(regex=feature_regex)
        cols = df.filter(regex="TARGET").columns.to_list()
        pbar = tqdm(cols, leave=True)
        for target in pbar:
            X = features.to_numpy()
            y = df[target]
            seq = StepwiseSelect(
                n_features_to_select=max_n_features,
                tol=1e-5,
                estimator=model.get(params),
                direction="forward",
                scoring=scoring,
                cv=5,
                n_jobs=-1,
                inner_progress=inner_progress,
            )
            seq.fit(X, y)
            s = np.array(seq.iteration_scores)
            best = np.max(seq.iteration_scores)
            if scoring in RegressionMetric.inverted():
                best = -best
            n_best = np.min(np.where(s == s.max()))
            info = seq.iteration_metrics[n_best]
            info = info.drop(columns=["fit_time", "score_time"])
            info = info.mean()
            for reg in RegressionMetric.inverted():
                info[reg.value] = -info[reg.value]
            if holdout is not None:
                idx = seq.iteration_features
                X_test = df_test.filter(regex=feature_regex).iloc[:, idx].to_numpy()
                y_test = df_test[target]
                estimator = model.get(params)
                estimator.fit(features.iloc[:, idx].to_numpy(), y)
                y_pred = estimator.predict(X_test)
                holdout_info = {
                    f"test_{reg.value}": reg(y_test, y_pred) for reg in RegressionMetric
                }
            else:
                holdout_info = {}
            all_scores.append(
                DataFrame(
                    {
                        "source": feature_regex,
                        "model": model.value,
                        "target": str(target).replace("TARGET__", ""),
                        "scorer": scoring.value,
                        "best_score": best,
                        **info.to_dict(),
                        **holdout_info,
                        "n_best": n_best,
                        "features": str(seq.iteration_features),
                    },
                    index=[count],
                )
            )
            name = str(target).replace("TARGET__", "")
            metric = (
                f"{np.round(100 * best, 3)}%"
                if scoring is RegressionMetric.ExplainedVariance
                else f"{np.round(best, 4)}"
            )
            pbar.set_description(
                f"{name}: {scoring.value.upper()}={metric} @ {n_best} features"
            )
            count += 1
    scores = pd.concat(all_scores, axis=0)
    scores.to_parquet(scores_out)
    print(f"Saved scores to {scores_out}")
    return scores


if __name__ == "__main__":
    # all_scores = []
    # for regex in ["CMC", "FS", "FS|CMC"]:
    #     scores = stepup_feature_select(
    #         regex=regex,
    #         model=RegressionModel.Lasso,
    #         scoring=RegressionMetric.MeanAbsoluteError,
    #         max_n_features=40,
    #         inner_progress=False,
    #         holdout=0.25,
    #         params=dict(alpha=10.0),
    #     )
    #     # scores["source"] = regex
    #     print(scores.drop(columns=["features"]))
    #     all_scores.append(scores)
    # df = pd.concat(all_scores, axis=0)

    # sys.exit()
    reduce_cmc = False
    reduce_targets = True
    df = FreesurferStatsDataset.HCP.load_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )
    sbn.set_style("darkgrid")
    targs = df.filter(regex="TARGET").rename(columns=lambda s: s.replace("TARGET__", ""))
    targs.hist(bins=50, color="black")
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.tight_layout()
    fig.savefig(str(PLOTS / "HCP_latent_target_distributions.png"), dpi=300)
    plt.close()

    print(
        targs.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975])
        .round(2)
        .T.to_markdown(tablefmt="simple", floatfmt="0.2f")
    )

    sys.exit()
    # cross_val_score(LGB(max_depth=1, min_data_in_leaf=5, extra_trees=True, max_bin=20, n_jobs=-1), x, y, cv=5, scoring="explained_variance")  # NOTE: for CMC

    scores = [stepup_feature_select(regex=reg) for reg in ["CMC", "FS", "FS|CMC"]]
    scores = pd.concat(scores, axis=0).reset_index(drop=True)
    scores.insert(3, "CMC ratio", 0.0)
    scores["Feat Names"] = ""
    for i in range(scores.shape[0]):
        regex = scores.loc[i, "source"]
        n = int(scores.loc[i, "n_best"])  # type: ignore
        idx = np.array(eval(scores.loc[i, "features"]), dtype=np.int64)
        cols = df.filter(regex=regex).columns.to_numpy()[idx]
        n_cmc = sum("CMC" in col for col in cols)
        if regex == "CMC":
            scores.loc[i, "CMC ratio"] = n_cmc / n
        scores.loc[i, "Feat Names"] = f'"{str(cols)}"'

    scores_out = TABLES / "stepup_selected_scores_linear_all_sources.csv.gz"
    scores.to_csv(scores_out, compression={"method": "gzip", "compresslevel": 9})
    print(f"Saved linear scores to {scores_out}")
    print(scores)
    sys.exit()
    pivoted = (
        scores.drop(columns="features")
        .groupby("source")
        .apply(
            lambda g: g.drop(columns="source").sort_values(
                by=["test_exp_var", "target"], ascending=[False, True]
            )
        )
        .droplevel(1)
        .reset_index()
        .pivot(columns="source", index="target")
    )
    print(
        pivoted.sort_values(
            by=[("Exp.Var", "FS|CMC"), ("Exp.Var", "FS"), ("Exp.Var", "CMC")],
            ascending=False,
        )
        .round(2)
        .rename(
            columns={
                "n_best": "n_selected",
                "Exp.Var": "Explained Variance (%)",
                "CMC ratio": "Proportion CMC Features",
            }
        )
    )

    # features = df.filter(regex=feature_regex)

    # pd.options.display.max_rows = 500
    # load_HCP_complete.clear()
