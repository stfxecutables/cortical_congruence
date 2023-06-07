from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

DATA = ROOT / "data"
ABIDE_I = DATA / "ABIDE-I"
ABIDE_II = DATA / "ABIDE-II"

ABIDE_I_PHENO = DATA / "ABIDE_I_Phenotypic_V1_0b.csv"
ABIDE_II_PHENO = DATA / "ABIDEII_Composite_Phenotypic.csv"

