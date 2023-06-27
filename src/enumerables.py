from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import Enum

from src.constants import DATA


class ROISource(Enum):
    Base = "base"
    Pial = "pial"
    DKT = "dkt"


class FreesurferStatsDataset(Enum):
    ABIDE_I = "abide-i"
    ABIDE_II = "abide-ii"
    ADHD_200 = "adhd-200"
    HBN = "hbn"
    HCP = "hcp"
    QTAB = "qtab"  # special, since two sessions
    QTIM = "qtim"  # special, since two sessions

    def root(self) -> Path:
        return {
            FreesurferStatsDataset.ABIDE_I: DATA / "ABIDE-I",
            FreesurferStatsDataset.ABIDE_II: DATA / "ABIDE-II",
            FreesurferStatsDataset.ADHD_200: DATA / "ADHD-200",
            FreesurferStatsDataset.HBN: DATA / "HBN",
            FreesurferStatsDataset.HCP: DATA / "HCP",
            FreesurferStatsDataset.QTAB: DATA / "QTAB",
            FreesurferStatsDataset.QTIM: DATA / "QTIM",
        }[self]

    def phenotypic_file(self) -> Path:
        root = self.root() / "phenotypic_data"

        return {
            FreesurferStatsDataset.ABIDE_I: root / "ABIDE_I_Phenotypic_V1_0b.csv",
            FreesurferStatsDataset.ABIDE_II: root / "ABIDEII_Composite_Phenotypic.csv",
            FreesurferStatsDataset.ADHD_200: root
            / "ADHD200_preprocessed_phenotypics.tsv",
            FreesurferStatsDataset.HBN: root / "HBN_R10_Pheno.csv",
            FreesurferStatsDataset.HCP: root / "unrestricted_behavioral.csv",
            FreesurferStatsDataset.QTAB: root / "phenotypic_data.tsv",
            FreesurferStatsDataset.QTIM: root / "participants.tsv",
        }[self]

    def has_pial_stats(self) -> bool:
        """
        Returns
        -------
        has_pial: bool
            Returns True if `*h.aparc.pial.stats` files exist for most subjects.

        """
        return {
            FreesurferStatsDataset.ABIDE_I: False,  # definitely not
            FreesurferStatsDataset.ABIDE_II: True,
            FreesurferStatsDataset.ADHD_200: True,
            FreesurferStatsDataset.HBN: True,
            FreesurferStatsDataset.HCP: False,  # does not seem to
            FreesurferStatsDataset.QTAB: False,  # not currently
            FreesurferStatsDataset.QTIM: False,  # not currently
        }[self]

    def has_dkt_atlas_stats(self) -> bool:
        """
        Returns
        -------
        has_stas: bool
            Returns True if `*h.aparc.DKTatlas.stats` files exist for most subjects.

        """
        return {
            FreesurferStatsDataset.ABIDE_I: False,  # definitely not
            FreesurferStatsDataset.ABIDE_II: True,
            FreesurferStatsDataset.ADHD_200: True,
            FreesurferStatsDataset.HBN: True,
            FreesurferStatsDataset.HCP: False,  # does not seem to
            FreesurferStatsDataset.QTAB: False,  # not currently
            FreesurferStatsDataset.QTIM: False,  # not currently
        }[self]
