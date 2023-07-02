from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import Enum
from pandas import DataFrame

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

    def freesurfer_files(self, ses2: bool = False) -> tuple[Path, Path] | Path | None:
        root = self.root()

        file = {
            FreesurferStatsDataset.ABIDE_I: None,
            FreesurferStatsDataset.ABIDE_II: None,
            FreesurferStatsDataset.ADHD_200: None,
            FreesurferStatsDataset.HBN: None,
            FreesurferStatsDataset.HCP: root / "unrestricted_hcp_freesurfer.csv",
            FreesurferStatsDataset.QTAB: None,
            FreesurferStatsDataset.QTIM: (
                (
                    root / "stats/fs5.3_ses-01.tsv",
                    root / "stats/fs5.3_subfields6_ses-01.tsv",
                ),
                (
                    root / "stats/fs5.3_ses-02.tsv",
                    root / "stats/fs5.3_subfields6_ses-02.tsv",
                ),
            ),
        }[self]
        if not isinstance(file, tuple):
            return file

        files = file[1] if ses2 else file[0]
        return files

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

    def load_stats_table(self) -> DataFrame:
        """Depending on data structure, either loop over stats files and build
        stats tables for each subject, or convert HCP and other pre-made table
        data to a common form, described below

        Notes
        -----
        Common table structure is as follows:

        """
        # TODO: save table in cached_results
        if self in [
            FreesurferStatsDataset.ABIDE_I,
            FreesurferStatsDataset.ABIDE_II,
            FreesurferStatsDataset.ADHD_200,
        ]:
            root = self.root()
        raise NotImplementedError()


class PhenotypicFocus(Enum):
    All = "all"
    Reduced = "reduced"
    Focused = "focused"

    def hcp_dict_file(self) -> Path:
        root = FreesurferStatsDataset.HCP.phenotypic_file().parent
        files = {
            PhenotypicFocus.All: root / "all_features.csv",
            PhenotypicFocus.Reduced: root / "features_of_interest.csv",
            PhenotypicFocus.Focused: root / "priority_features_of_interest.csv",
        }
        if self not in files:
            raise ValueError(f"Invalid {self.__class__.__name__}: {self}")
        return files[self]

    def desc(self) -> str:
        return {
            PhenotypicFocus.All: "all",
            PhenotypicFocus.Reduced: "plausible",
            PhenotypicFocus.Focused: "most sound",
        }[self]
