from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from argparse import Namespace
from math import ceil
from pathlib import Path
from random import shuffle
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas import DataFrame
from sklearn.dummy import DummyRegressor as Dummy
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import cross_validate, train_test_split
from tqdm import tqdm

from src.constants import TABLES
from src.enumerables import PhenotypicFocus, RegressionMetric, RegressionModel
from src.feature_selection.stepwise import StepwiseSelect
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
    rand = args.random_feats
    perc_max = args.perc_max
    x = dfr.filter(regex=args.feature_regex)
    if rand:
        idx = np.random.permutation(x.shape[1])
        n = np.random.randint(5, max(6, ceil(perc_max * x.shape[1])))
        x = x.iloc[:, idx].iloc[:, :n]
    else:
        n = x.shape[1]

    x = x.to_numpy()
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
    df.drop(columns=["fit_time", "score_time"], inplace=True)
    df.loc[:, "MAE"] = df["MAE"].abs()
    df.loc[:, "R2"] = df["R2"].abs()
    df.loc[:, "Exp.Var"] = df["Exp.Var"].abs()
    df["target"] = str(target).replace("TARGET__", "")
    df["model"] = modelname
    df["n_feats"] = n
    return df


def get_random_subset_args(
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
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
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
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
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


def sample_random_subsets(use_mean: bool) -> None:
    summaries = []
    for regex in ["FS", "CMC", "FS|CMC"]:
        reduce_cmc = False
        feature_regex = regex
        random_feats = True
        n_random = 5000
        perc_max = 0.3

        args = get_random_subset_args(
            feature_regex=feature_regex,
            reduce_cmc=reduce_cmc,
            random_feats=random_feats,
            n_random=n_random,
            perc_max=perc_max,
        )
        r = "reduced" if reduce_cmc else "raw"
        f = feature_regex.lower()
        if "|" in r:
            r = "+".join(sorted(r.split("|")))
        rnd = f"_rand_{n_random}_of{perc_max:0.2f}p" if random_feats else ""
        out = TABLES / f"quick_regression_cluster_targets_{f}_{r}{rnd}.parquet"
        if out.exists():
            df = pd.read_parquet(out)
        else:
            print(f"Computing {len(args)} iterations:")
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
            .drop_duplicates()
        )
        base_mae = df.target.apply(lambda target: dummy_maes[target])
        df.insert(3, "sMAE", df["MAE"] / base_mae)
        best = df[df["sMAE"] < 1.0]
        print(
            best.sort_values(by="sMAE", ascending=True)
            .iloc[:200]
            .sort_values(by="sMAE", ascending=False)
            .to_markdown(tablefmt="simple", index=False, floatfmt="0.4f")
        )
        target_bests = (
            df[["target", "Exp.Var", "sMAE", "n_feats"]]
            .groupby("target")
            .apply(lambda g: g.nsmallest(10 if use_mean else 1, columns=["sMAE"]))
            .droplevel(1)
            .drop(columns="target")
            .sort_values(by=["target", "sMAE"])
        )
        if use_mean:
            target_bests = (
                target_bests.groupby("target").mean().sort_values(by=["target", "sMAE"])
            )

        print("=" * 90)
        print(out.name)
        print("=" * 90)
        print(
            target_bests.to_markdown(
                tablefmt="simple", floatfmt=["", "0.5f", "0.4f", "0.0f"]
            )
        )

        target_bests["Exp.Var"] = target_bests["Exp.Var"] * 100
        target_bests = target_bests.rename(columns={"Exp.Var": "Exp.Var (%)"})
        label = regex.replace("|", "+")
        summaries.append(target_bests.rename(columns=lambda s: f"{label}_{s}"))

    summary = pd.concat(summaries, axis=1)
    order = [
        "FS_Exp.Var (%)",
        "CMC_Exp.Var (%)",
        "FS+CMC_Exp.Var (%)",
        "FS_sMAE",
        "CMC_sMAE",
        "FS+CMC_sMAE",
        "FS_n_feats",
        "CMC_n_feats",
        "FS+CMC_n_feats",
    ]
    summary = summary[order].copy()
    for col in summary.filter(regex="n_feats").columns:
        summary[col] = summary[col].astype(str)

    summary = summary.sort_values(
        by=["FS_sMAE", "CMC_sMAE", "FS+CMC_sMAE"], ascending=True
    )
    print(
        summary.to_markdown(
            tablefmt="simple",
            floatfmt=[
                "",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.3f",
                "0.1f",
                "0.1f",
                "0.1f",
            ],
        )
    )
    summary_out = TABLES / "random_subset_summary.csv.gz"
    summary.to_csv(summary_out, compression={"method": "gzip", "compresslevel": 9})
    print(f"Saved feature subset summary to {summary_out}")


def stepup_feature_select(
    regex: str,
    model: RegressionModel,
    scoring: RegressionMetric = RegressionMetric.ExplainedVariance,
    max_n_features: int = 100,
    inner_progress: bool = False,
    holdout: float | None = None,
) -> DataFrame:
    h = f"_holdout={holdout}" if holdout is not None else ""
    fname = (
        f"stepup_{scoring.value}_selected"
        f"_{model.value}_{regex}_n={max_n_features}{h}.parquet"
    )
    scores_out = TABLES / fname
    if scores_out.exists():
        return pd.read_parquet(scores_out)

    reduce_cmc = False
    reduce_targets = True
    df = load_HCP_complete(
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
                estimator=model.get(),
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
                estimator = model.get()
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
    scores = stepup_feature_select(
        regex="CMC",
        model=RegressionModel.Ridge,
        scoring=RegressionMetric.MeanAbsoluteError,
        max_n_features=100,
        inner_progress=False,
        holdout=0.25,
    )
    print(scores)

    sys.exit()
    reduce_cmc = False
    reduce_targets = True
    df = load_HCP_complete(
        focus=PhenotypicFocus.All, reduce_targets=reduce_targets, reduce_cmc=reduce_cmc
    )

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
                by=["Exp.Var", "target"], ascending=[False, True]
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
