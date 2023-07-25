from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

from src.constants import MEMORY
from src.enumerables import ClassificationMetric, FreesurferStatsDataset


@MEMORY.cache
def get_dummy_perfs() -> tuple[DataFrame, DataFrame]:
    """
    Second return value:

    | data     | target               |   acc |   acc_bal |    f1 |   precision |   recall |    N |
    |:---------|:---------------------|------:|----------:|------:|------------:|---------:|-----:|
    | abide-i  | CLS__autism          | 0.563 |     0.563 | 0.682 |       0.563 |    0.563 | 1108 |
    | abide-i  | CLS__dsm_iv_spectrum | 0.551 |     0.554 | 0.7   |       0.554 |    0.554 | 1036 |
    | abide-i  | MULTI__dsm_iv        | 0.538 |     0.346 | 0.7   |       0.538 |    0.346 | 1036 |
    | abide-ii | CLS__autism          | 0.586 |     0.574 | 0.71  |       0.578 |    0.574 |  698 |
    | abide-ii | CLS__dsm5_spectrum   | 0.575 |     0.551 | 0.73  |       0.575 |    0.551 |  434 |
    | abide-ii | MULTI__dsm_iv        | 0.575 |     0.38  | 0.73  |       0.575 |    0.38  |  434 |
    | adhd-200 | CLS__adhd_spectrum   | 0.594 |     0.599 | 0.72  |       0.599 |    0.599 |  321 |
    | adhd-200 | MULTI__diagnosis     | 0.571 |     0.381 | 0.727 |       0.571 |    0.381 |  316 |

    """
    strategies = ["prior", "stratified", "uniform"]
    scorers = ClassificationMetric.scorers()

    all_results = []
    for ds in [
        FreesurferStatsDataset.ABIDE_I,
        FreesurferStatsDataset.ABIDE_II,
        FreesurferStatsDataset.ADHD_200,
        FreesurferStatsDataset.HCP,
    ]:
        df = ds.load_complete()
        cols = (
            df.filter(regex="CLS").columns.to_list()
            + df.filter(regex="MULTI").columns.to_list()
        )
        for target_col in cols:
            y = df[target_col].dropna()
            for strat in strategies:
                model = DummyClassifier(strategy=strat)

                results = cross_validate(
                    model,
                    y,
                    y,
                    cv=5,
                    scoring=scorers,
                    n_jobs=5,
                    error_score="raise",
                )
                result = DataFrame(results)
                result.insert(0, "data", ds.value)
                result.insert(0, "target", str(target_col).replace("TARGET__", ""))
                result.insert(0, "strat", strat)
                result["N"] = len(y)
                all_results.append(
                    result.rename(columns=lambda s: s.replace("test_", "")).drop(
                        columns=["fit_time", "score_time"]
                    )
                )
    df = pd.concat(all_results, axis=0, ignore_index=True)
    max_by_strat = df.groupby(["data", "target", "strat"]).max()
    cross_strat_maxes = df.drop(columns="strat").groupby(["data", "target"]).max()
    return max_by_strat.reset_index(), cross_strat_maxes.reset_index()


if __name__ == "__main__":
    pd.options.display.max_rows = 500
    max_by_strat, cross_strat_maxes = get_dummy_perfs()
    print(max_by_strat.round(3))
    print(cross_strat_maxes.round(3).to_markdown())
