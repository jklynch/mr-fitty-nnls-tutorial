"""The bundled arsenic XANES spectra, as a reference matrix R and a measured spectrum s."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "arsenic"
UNKNOWN = "OTT3_55_spot0"

# Short labels for the plots. The file names are what the beamline and Athena called them.
SHORT_NAMES = {
    "Arsenopyrite_Julcani_OA": "arsenopyrite",
    "orpiment_all_ref_als_cal": "orpiment",
    "arsenate_sorbed_diop_avg_als_cal": "arsenate on diopside",
    "As2O3_ref_avg_als_cal": "As$_2$O$_3$",
    "sodium_arsenite_13BM_ref_als_cal": "sodium arsenite",
    "As_pyrite_A_Foster_sln": "As-pyrite",
    "Lollingite_Lolling_avg_OA": "löllingite",
    "scorodite_A_Foster_sln": "scorodite",
    "arsenite_aqueous_als_cal": "aqueous arsenite",
    "sodium_arsenate_ref_als_cal": "sodium arsenate",
    "OTT3_55_spot0": "OTT3_55 spot 0",
}


def read_athena(path):
    """(energy, norm) from an Athena .e export: '#' header lines, then energy and norm columns."""
    table = np.loadtxt(path, comments="#")
    return table[:, 0], table[:, 1]


def reference_names():
    """Every bundled reference, by file stem, sorted."""
    return sorted(p.stem for p in (DATA_DIR / "reference").glob("*.e"))


@dataclass
class Spectra:
    """One unknown and some references on a common energy grid.

    R[:, i] is reference r_i at the unknown's energies, so fitting s by R w is asking how much of
    each reference the unknown contains.
    """

    energies: np.ndarray  # (n,) eV
    R: np.ndarray  # (n, m) one reference per column
    s: np.ndarray  # (n,) the unknown
    names: list  # m reference file stems
    unknown: str

    @property
    def labels(self):
        return [SHORT_NAMES.get(name, name) for name in self.names]


def load_arsenic(references=None, unknown=UNKNOWN, energy_range=None):
    """The unknown and the chosen references, references interpolated onto the unknown's energies.

    Only the unknown's energies that every reference also covers are kept, so nothing is
    extrapolated. References are resampled with an interpolating cubic spline, as MrFitty does.
    `energy_range` (lo, hi) in eV optionally narrows the grid further.
    """
    names = list(references) if references is not None else reference_names()
    energies, s = read_athena(DATA_DIR / "unknown" / f"{unknown}.e")
    curves = [read_athena(DATA_DIR / "reference" / f"{name}.e") for name in names]
    lo = max(e[0] for e, _ in curves)
    hi = min(e[-1] for e, _ in curves)
    if energy_range is not None:
        lo, hi = max(lo, energy_range[0]), min(hi, energy_range[1])
    keep = (energies >= lo) & (energies <= hi)
    energies, s = energies[keep], s[keep]
    R = np.column_stack([make_interp_spline(e, y, k=3)(energies) for e, y in curves])
    return Spectra(energies=energies, R=R, s=s, names=names, unknown=unknown)
