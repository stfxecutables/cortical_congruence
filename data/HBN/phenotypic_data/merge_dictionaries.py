from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import sys
from pathlib import Path
from warnings import filterwarnings, warn

import numpy as np
import pandas as pd
from pandas import DataFrame

DICTS_ROOT = Path(__file__).resolve().parent / "Release11_DataDic"
DATA_DICTS = sorted(DICTS_ROOT.glob("*.xlsx"))

RENAMES = {
    "Item": "item",
    "Subtest": "item",
    "Question": "item",
    "Variable": "name",
    "Variable Name": "name",
    "Variable Type": "type",
    "Values": "range",
    "Value": "range",
    "Response Values": "categorical_labels",
    "Value Labels": "categorical_labels",
    "Value Scale": "categorical_labels",
    "Value Label": "categorical_labels",
}


def parse_table(path: Path) -> DataFrame | None:
    name = path.name
    # junk scales not worth parsing / fixing
    if path.stem in ["ESWAN", "MRI_Track", "SympChck"]:
        return None
    # if name != "Diagnosis_KSADS.xlsx":
    #     return
    table = pd.read_excel(path, usecols=[0, 1, 2, 3])

    scale_title = table.columns.to_list()[0].strip()
    table.columns = table.iloc[0, :].values
    table = table.rename(columns=str.strip).rename(
        columns=RENAMES,
    )
    potential_drop_idxs = table[table["name"].isna()].index.to_list()
    drop_idxs = []
    header_idxs = []
    headers = []
    score_idxs = []
    for idx in potential_drop_idxs:
        if "Scores" in str(table.loc[idx, "item"]):
            drop_idxs.append(idx)
            score_idxs.append(idx)
        elif table.loc[idx, ["name"]].item() is np.nan:
            drop_idxs.append(idx)
            headers.append(str(table.loc[idx, "item"]).strip())
            header_idxs.append(idx)
        else:
            raise RuntimeError(f"Found anomalous row in {name}:\n{table.loc[idx]}")
    if len(headers) > 1:
        warn(f"Found multiple headers in {name}")

    table = table.drop(index=[0] + drop_idxs)
    for col in table:  # can't trust these kinds of datasets
        table[col] = table[col].str.strip()

    expected_cols = sorted(np.unique(list(RENAMES.values())).tolist())
    found_cols = sorted(table.columns.to_list())
    if len(found_cols) > len(expected_cols):
        raise RuntimeError(
            f"Too many column(s) in {path}.\nExpected:\n{expected_cols}.\nGot:\n{found_cols}"
        )
    if len(found_cols) < len(expected_cols):
        missing = sorted(list(set(expected_cols).difference(found_cols)))
        for col in missing:
            table[col] = np.nan
    found_cols = sorted(table.columns.to_list())
    if found_cols != expected_cols:
        raise RuntimeError(
            f"Misnamed column(s) in {path}.\nExpected:\n{expected_cols}.\nGot:\n{found_cols}"
        )

    table["scale_name"] = scale_title

    print(name)
    print(table)

    return table


if __name__ == "__main__":
    filterwarnings("ignore", "Unknown extension.*", UserWarning)
    tables = []
    for p in DATA_DICTS:
        try:
            df = parse_table(p)
            if df is not None:
                tables.append(df)
        except RuntimeError as e:
            print(f"Got error for dict at {p}")
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to parse table at {p}") from e
    df = pd.concat(tables, axis=0, ignore_index=True)
    out = Path(__file__).resolve().parent / "data_dict_all.parquet"
    df.to_parquet(out)
    print(f"Saved complete data dictionary to {out}")
