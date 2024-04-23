from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import numpy as np  # isort: skip

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
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import lightgbm as lgbm
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from optuna import Study, Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from optuna_integration.lightgbm import LightGBMPruningCallback, LightGBMTunerCV
from pandas import DataFrame, Index, Series
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.nn import Dropout, HuberLoss, LeakyReLU, Linear, Module, Parameter, Sequential
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchmetrics import ExplainedVariance, MeanAbsoluteError
from torchmetrics.functional import explained_variance
from typing_extensions import Literal

DATA = ROOT / "data/complete_tables"
HCP = (
    DATA / "HCP/HCP_all_FS_all_CMC_all_phenotypic_reduced_TARGET__REG__int_g_like.parquet"
)
TARGET = "TARGET__REG__int_g_like"
FIXED_ARGS = dict(verbosity=-1)
DEFAULT_ARGS = dict(bagging_freq=1, bagging_fraction=0.75, n_jobs=1)


def normalize(df: DataFrame, target: str | None = None, robust: bool = True) -> DataFrame:
    """
    Clip data to within twice of its "robust range" (range of 90% of the data),
    and then min-max normalize.
    """
    if (target in df.columns) and (target is not None):
        X = df.drop(columns=target)
    else:
        X = df

    cols = X.columns

    # need to not normalize one-hot columns...

    if robust:
        medians = np.nanmedian(X, axis=0)
        X = X - medians  # robust center

        # clip values 2 times more extreme than 95% of the data
        rmins = np.nanpercentile(X, 5, axis=0)
        rmaxs = np.nanpercentile(X, 95, axis=0)
        rranges = rmaxs - rmins
        rmins -= 2 * rranges
        rmaxs += 2 * rranges
        X = np.clip(X, a_min=rmins, a_max=rmaxs)

    X_norm = DataFrame(data=MinMaxScaler().fit_transform(X), columns=cols)
    if (target in df.columns) and (target is not None):
        X_norm = pd.concat([X, df[target]], axis=1)
    return X_norm


class LinearSelectorModel(Module):
    def __init__(self, in_features: int, n_select: int = 20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_feat = in_features
        self.n_select = n_select
        noise = torch.randn([1, self.n_select, in_features]) / self.n_feat
        init = torch.full([1, self.n_select, in_features], 1 / self.n_feat)
        w = init + noise
        self.alpha = Parameter(w, requires_grad=True)
        self.model = Sequential(
            Linear(in_features=self.n_feat * self.n_select, out_features=512),
            LeakyReLU(),
            # Linear(in_features=512, out_features=1),
            Linear(in_features=512, out_features=512),
            LeakyReLU(),
            Dropout(0.2),
            Linear(in_features=512, out_features=256),
            LeakyReLU(),
            Linear(in_features=256, out_features=64),
            LeakyReLU(),
            Linear(in_features=64, out_features=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, 1, in_features)
        alpha = torch.softmax(self.alpha, dim=-1)
        x = x * alpha
        x = x.flatten(1)
        x = self.model(x)
        return x


class LinearSelector(LightningModule):
    def __init__(
        self,
        in_features: int,
        n_select: int,
        lr: float = 3e-4,
        wd: float = 1e-4,
        n_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.n_epochs = n_epochs
        self.model = LinearSelectorModel(in_features=in_features, n_select=n_select)
        self.criterion = HuberLoss()
        self.mae = MeanAbsoluteError()
        self.var_exp = ExplainedVariance()

    def training_step(
        self, batch: tuple[Tensor, Tensor], *args: Any, **kwargs: Any
    ) -> Tensor:
        return self.shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], *args: Any, **kwargs: Any
    ) -> Tensor:
        return self.shared_step(batch, "val")

    def shared_step(self, batch: tuple[Tensor, Tensor], phase: str) -> Tensor:
        x, target = batch
        preds = self.model(x).squeeze()
        loss = self.criterion(preds, target)
        with torch.no_grad():
            mae = self.mae(preds, target)
            varexp = self.var_exp(preds, target)
            self.log(f"{phase}/mae", mae, prog_bar=True)
            self.log(f"{phase}/var.exp", varexp, prog_bar=True)
            self.log(f"{phase}/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=self.n_epochs)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def get_loaders(
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataFrame, Series, DataFrame, Series, int, Index[str]]:
    df = pd.read_parquet(HCP)
    x = df.drop(columns=TARGET)
    cols = x.columns
    x = normalize(x)
    x = SimpleImputer(strategy="median").fit_transform(x)
    y = df[TARGET]
    y = MinMaxScaler().fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

    x, x_test, y, y_test = train_test_split(x, y, test_size=200, shuffle=True)
    df_x_train = DataFrame(data=x, columns=cols)
    df_y_train = Series(data=y, name=df[TARGET].index.name)
    x_test = DataFrame(data=x_test, columns=cols)
    y_test = Series(data=y_test, name=df[TARGET].index.name)

    x = torch.tensor(data=x.copy(), dtype=torch.float, device="cpu").unsqueeze(1)
    y = torch.tensor(data=y.copy(), dtype=torch.float, device="cpu")

    ds = TensorDataset(x, y)
    ds_train, ds_val = random_split(ds, lengths=[0.3, 0.7])
    ds_args: Mapping[str, Any] = dict(
        batch_size=batch_size,
        persistent_workers=True,
        drop_last=True,
        num_workers=4,
    )
    train = DataLoader(ds_train, shuffle=True, **ds_args)
    val = DataLoader(ds_val, shuffle=False, **ds_args)
    n_feat = x.shape[-1]
    return train, val, df_x_train, df_y_train, x_test, y_test, n_feat, cols


OPT_LOGGER = optuna.logging._get_library_root_logger()


class EarlyStopping:
    def __init__(self, patience: int = 10, min_trials: int = 50) -> None:
        self.patience: int = patience
        self.min_trials: int = min_trials
        self.has_stopped: bool = False

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        """https://github.com/optuna/optuna/issues/1001#issuecomment-1351766030"""
        if self.has_stopped:
            # raise optuna.exceptions.TrialPruned()
            study.stop()

        current_trial = trial.number
        if current_trial < self.min_trials:
            return
        best_trial = study.best_trial.number

        # best_score = study.best_value  # TODO: patience
        should_stop = (current_trial - best_trial) >= self.patience
        if should_stop:
            if not self.has_stopped:
                OPT_LOGGER.info(
                    f"Completed {self.patience} trials without metric improvement. "
                    f"Stopping early at trial {current_trial}. Some trials may still "
                    "need to finish, and produce a better result. If so, that better "
                    f"result will be used rather than trial {current_trial}."
                )
            self.has_stopped = True
            study.stop()


def optuna_args(trial: Trial) -> dict[str, str | float | int]:
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 300, step=50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 256, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 7),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
    )


def optuna_objective(
    X_train: DataFrame,
    y_train: Series,
    metric: Literal["mae"] | Literal["exp.var"],
    n_folds: int = 3,
) -> Callable[[Trial], float]:
    X = np.asarray(X_train)
    y = np.asarray(y_train)

    def objective(trial: Trial) -> float:
        kf = KFold
        _cv = kf(n_splits=n_folds, shuffle=True)
        opt_args = optuna_args(trial)
        full_args = {**FIXED_ARGS, **DEFAULT_ARGS, **opt_args}
        scores = []
        for step, (idx_train, idx_test) in enumerate(_cv.split(X_train, y_train)):
            X_tr, y_tr = X[idx_train], y[idx_train]
            X_test, y_test = X[idx_test], y[idx_test]
            # model_cls, clean_args = model_cls_args(full_args)
            estimator = LGBMRegressor(**full_args)
            estimator.fit(X_tr, y_tr)
            preds = estimator.predict(X_test)

            score = (
                explained_variance(torch.from_numpy(y_test), torch.from_numpy(preds))
                if metric == "exp.var"
                else mean_absolute_error(y_test, np.asarray(preds))
            )
            scores.append(score)
            # allows pruning
            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    return objective


def tune_lgbm(
    X_train: DataFrame,
    y_train: Series,
    X_test: DataFrame,
    y_test: Series,
    metric: Literal["mae"] | Literal["exp.var"],
) -> None:
    direction = "maximize" if metric == "exp.var" else "minimize"
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(),
        pruner=MedianPruner(n_warmup_steps=10, n_min_trials=50),
    )
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    objective = optuna_objective(X_train=X_train, y_train=y_train, metric=metric)
    cbs = [EarlyStopping(patience=15, min_trials=100)]
    cbs = []
    study.optimize(
        objective,
        n_trials=200,
        callbacks=cbs,
        timeout=300,
        n_jobs=-1,
        gc_after_trial=False,
        show_progress_bar=False,
    )
    tuned_args = {**FIXED_ARGS, **DEFAULT_ARGS, **study.best_params}
    model = LGBMRegressor(**tuned_args)
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test))
    scorer = (
        mean_absolute_error
        if metric == "mae"
        else lambda preds, y_test: explained_variance(
            torch.from_numpy(preds), torch.from_numpy(y_test.to_numpy())
        )
    )
    score = scorer(preds, y_test)
    name = "MAE" if metric == "mae" else "Var.Exp"
    print(f"Tuned {name}: {score}")


def get_alpha() -> ndarray: ...


if __name__ == "__main__":
    BATCH = 32
    EPOCHS = 20
    LR = 3e-3
    WD = 1e-1
    N_SELECT = 20
    train, val, X_tr, y_tr, X_test, y_test, n_feat, cols = get_loaders(batch_size=BATCH)
    trainer = Trainer(
        accelerator="mps", max_epochs=EPOCHS, callbacks=TQDMProgressBar(), logger=False
    )
    model = LinearSelector(
        in_features=n_feat, n_select=N_SELECT, n_epochs=EPOCHS, lr=LR, wd=WD
    )

    trainer.fit(model, train, val)
    model.eval()
    alpha = model.model.alpha
    a = alpha.clone().detach().cpu().numpy().squeeze()  # (N_select, n_feat)
    feat_idx = np.unique(
        torch.topk(alpha.detach().squeeze(), dim=1, k=3)[1].ravel().numpy()
    )
    # feat_idx = np.unique(np.argmax(a, axis=1))
    selected = [cols[idx] for idx in feat_idx]

    print("Selected:", selected)
    X_tr = X_tr[selected]
    X_test = X_test[selected]

    # params = {
    #     "objective": "regression",
    #     "metric": "huber",
    #     "verbosity": -1,
    #     "boosting_type": "gbdt",
    # }
    # dtrain = lgbm.Dataset(X_tr, label=y_tr)
    # dtest = lgbm.Dataset(X_test, label=y_test)
    # tuner = LightGBMTunerCV(
    #     params,
    #     dtrain,
    #     folds=KFold(n_splits=3),
    #     callbacks=[lgbm.early_stopping(100), lgbm.log_evaluation(100)],
    # )
    # tuner.run()
    # print("Best score:", tuner.best_score)
    # best_params = tuner.best_params
    # print("Best params:", best_params)

    tune_lgbm(X_tr, y_tr, X_test, y_test, metric="exp.var")
    print()
