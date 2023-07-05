from typing import Mapping

import numpy as np
from numpy import ndarray
from sklearn.metrics import explained_variance_score as expl_var
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mad
from sklearn.metrics import r2_score as r2


def smae(y_true: ndarray, y_pred: ndarray, **kwargs: Mapping) -> float:
    y_dummy = np.abs(np.mean(y_true) - y_true)
    return float(mae(y_true, y_pred) / np.mean(y_dummy))


def smad(y_true: ndarray, y_pred: ndarray, **kwargs: Mapping) -> float:
    y_dummy = np.abs(np.median(y_true) - y_true)
    return float(mad(y_true, y_pred) / np.median(y_dummy))


mae_scorer = make_scorer(mae, greater_is_better=False)
smae_scorer = make_scorer(smae, greater_is_better=False)
mad_scorer = make_scorer(mad, greater_is_better=False)
smad_scorer = make_scorer(smad, greater_is_better=False)
expl_var_scorer = make_scorer(expl_var, greater_is_better=True)
r2_scorer = make_scorer(r2, greater_is_better=True)