"""Constrained stoichiometric model of self-purging pyrolysis.

Estimates the temperature-resolved mass fractions of the solid, liquid and
gaseous streams, and the composition of both condensed phases, from proximate
and ultimate analyses, thermogravimetry and gas chromatography. Atomic
conservation is enforced identically by analytical elimination; physical
admissibility is enforced by inequality constraints on the whole temperature
grid simultaneously. Measurement uncertainty is propagated by Monte Carlo.

Outputs are written as CSV files consumed by the LaTeX manuscript and by the
standalone TikZ figure sources.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Input data ship with the package; outputs are written to the working directory unless redirected
# through the environment variables below.
PACKAGE_DATA = Path(__file__).resolve().parent / "data"
DATA = Path(os.environ.get("CCPYRO_DATA", PACKAGE_DATA))
RESULTS = Path(os.environ.get("CCPYRO_RESULTS", Path.cwd() / "results"))
FIGDATA = Path(os.environ.get("CCPYRO_FIGDATA", Path.cwd() / "figdata"))
ROOT = Path.cwd()

T_REF = 200.0
T_NODES = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
N_NODES = T_NODES.size

# The number of draws can be reduced through CCPYRO_MC_DRAWS for a quick check of the whole chain.
N_MONTE_CARLO = int(os.environ.get("CCPYRO_MC_DRAWS", 300))
# Independent admissible draws added to the pool in build_cases, so that the
# reported estimate is the best of a multi-start and not the state a
# sequential solve happened to select.
RANDOM_STARTS = 0
RANDOM_SEED = 20250724

ALPHA_TRANSFERRED = 1.19
ALPHA_SENSITIVITY = (0.3, 0.5, 0.8, 1.19, 1.6, 2.2)
KAPPA_PRIOR = 0.85
AMPLITUDE_PRIOR = 0.45

M = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007}
MM = {
    "CO": 28.010,
    "CO2": 44.009,
    "CH4": 16.043,
    "H2": 2.016,
    "H2O": 18.015,
}
GAS_SPECIES = ("CO", "CO2", "CH4", "H2")
MM_GAS = np.array([MM[s] for s in GAS_SPECIES])

LHV_VOLUMETRIC = {"CO": 12.64, "H2": 10.80, "CH4": 35.80, "CO2": 0.0}

# Lump representing the hydrocarbons of two and three carbon atoms that the
# chromatographic method does not quantify. The formula is the mole-weighted mean
# of the C2H4, C2H6, C3H6 and C3H8 distribution reported for slow pyrolysis of
# herbaceous residues, expressed per two carbon atoms, and the volumetric heating
# value is the mole-weighted mean of the same distribution. Both are used only in
# the omitted-species error budget and never in the reported base case.
C2_CARBON = 2.0
C2_HYDROGEN = 4.60
MM_C2 = M["C"] * C2_CARBON + M["H"] * C2_HYDROGEN
LHV_C2 = 64.0  # MJ/Nm3, mole-weighted mean of the same distribution
# Mole ratio of the C2+ lump to methane in the released gas, by node. Only the
# values at 300 and 500 degrees Celsius rest on measurements of a comparable
# feedstock and reactor; the others are bounded extrapolations, and the low and
# high scenarios carry that extrapolation rather than a measurement dispersion.
C2_TO_METHANE = {
    "low": np.array([0.10, 0.15, 0.25, 0.20, 0.12, 0.08]),
    "central": np.array([0.24, 0.30, 0.40, 0.38, 0.26, 0.18]),
    "high": np.array([0.35, 0.50, 0.60, 0.55, 0.45, 0.35]),
}
VAPORISATION = 2.442          # MJ/kg, latent heat of water at 298.15 K
WATER_PER_HYDROGEN = MM["H2O"] / MM["H2"]
MOLAR_VOLUME = 22.414e-3  # Nm3/mol at 273.15 K and 101.325 kPa

# The experiment covers three pelletised feedstocks: rice husk, corncob and
# their equal-mass blend. The charge is always densified, so densification is a
# fixed condition of the study rather than a factor. A case is still identified
# by the pair (feedstock, densification) so that the labels of the result files
# name the physical form of the charge explicitly.
FEEDSTOCKS = ("rice husk", "blend", "corncob")
DENSIFICATIONS = ("pelletized",)
GAS_LABEL = {"rice husk": "RH100", "blend": "CC50:RH50", "corncob": "CC100"}
GAS_DENSIFICATION = {"pelletized": "Pelletized"}
CASES = (
    ("rice husk", "pelletized"),
    ("blend", "pelletized"),
    ("corncob", "pelletized"),
)
# Cell and temperature at which one bag was injected repeatedly. It is the only
# repeated measurement of the chromatographic campaign.
REPEATED_INJECTIONS = (("blend", "pelletized"), 800.0)
TREATMENTS = CASES
# Mass fraction of rice husk in the blend, as received.
BLEND_SHARE = 0.5
# Thermograms of a former campaign, recorded at 5 K/min, retained only for
# the heating-rate sensitivity of the results.
THERMOGRAM_FILE_5K = {
    "rice husk": "TGA_rice_husk.csv",
    "corncob": "TGA_corncob.csv",
}
THERMOGRAM_FILE = THERMOGRAM_FILE_5K
THERMOGRAM_30K = "TGA_30Cmin.csv"


def case_name(case: tuple[str, str]) -> str:
    return f"{case[0]} ({case[1]})"

VARIABLES = ("n_c", "x_c", "y_c", "x_t", "y_t", "n_CO", "n_CO2", "n_CH4")
N_FREE = len(VARIABLES)
IDX = {name: i for i, name in enumerate(VARIABLES)}
GLOBAL_PARAMETERS = ("kappa", "A", "a_H", "a_O")

PRIOR_MEAN = np.array([KAPPA_PRIOR, AMPLITUDE_PRIOR, 1.05, 1.30])
PRIOR_SD = np.array([0.85, 0.25, 0.30, 0.35])
PARAM_LOWER = np.array([0.02, 0.05, 0.10, 0.10])
PARAM_UPPER = np.array([8.00, 0.98, 3.00, 3.00])

VAR_LOWER = np.array([0.0, 0.02, 0.004, 0.45, 0.005, 0.0, 0.0, 0.0])
VAR_UPPER_TEMPLATE = np.array([1.0, None, None, None, None, 0.9, 0.9, 0.9])

SIGMA = {
    "tga": 0.010,
    "tga_floor": 0.005,
    "gc_floor": 0.005,
    "water": 0.100,
    "tar": 0.060,
    "van_krevelen": 0.150,
    "regularization": 0.050,
}
OMEGA = {
    "tga": 3.0,
    "gc": 1.0,
    "water": 1.0,
    "tar": 2.0,
    "van_krevelen": 1.0,
    "regularization": 1.0,
    "prior": 1.0,
}
ANCHORS = {
    "x_t_first_sd": 0.12,
    "y_t_first_sd": 0.10,
    "x_c_last": 0.30,
    "x_c_last_sd": 0.12,
    "y_c_last": 0.07,
    "y_c_last_sd": 0.04,
    "x_t_last": 0.95,
    "x_t_last_sd": 0.30,
    "y_t_last": 0.15,
    "y_t_last_sd": 0.10,
}
OBJECTIVE_SCALE = 100.0

# Repeatability of the characterisation, as standard deviations in absolute
# percentage points on the as-received basis. Proximate values follow the
# repeatability limits of ISO 18122, ISO 18123 and ISO 18134-3; the elemental
# values follow the specification of the analyser.
SIGMA_ANALYSIS = {
    "carbon": 0.50,
    "hydrogen": 0.15,
    "nitrogen": 0.05,
    "ash": 0.20,
    "moisture": 0.20,
    "fixed_carbon": 0.50,
}
SIGMA_THETA_RELATIVE = 0.015
# Absolute dispersion of the part of the by-difference carbon dioxide that does
# not cancel when the composition is renormalised over the four species, namely
# the argon of the entrained air and the shortfall of the oxygen channel at low
# concentration. The value is the bound derived in Section S5 of the
# Supplementary material for a bag holding a tenth of its volume as air.
SIGMA_AIR_RESIDUAL = 0.005


@dataclass(frozen=True)
class Feedstock:
    """Elemental and proximate description of one treatment."""

    label: str
    moisture: float
    ash: float
    volatile_matter: float
    fixed_carbon: float
    carbon: float
    hydrogen: float
    nitrogen: float

    @property
    def oxygen(self) -> float:
        return 100.0 - self.carbon - self.hydrogen - self.nitrogen - self.ash

    @property
    def organic_hydrogen(self) -> float:
        return self.hydrogen - self.moisture * 2.0 * M["H"] / MM["H2O"]

    @property
    def organic_oxygen(self) -> float:
        return self.oxygen - self.moisture * M["O"] / MM["H2O"]

    @property
    def daf_mass(self) -> float:
        return 100.0 - self.moisture - self.ash

    @property
    def closure_residual(self) -> float:
        total = self.carbon + self.organic_hydrogen + self.nitrogen + self.organic_oxygen
        return total - self.daf_mass

    @property
    def daf_composition(self) -> dict[str, float]:
        scale = 100.0 / self.daf_mass
        return {
            "C": self.carbon * scale,
            "H": self.organic_hydrogen * scale,
            "O": self.organic_oxygen * scale,
            "N": self.nitrogen * scale,
        }

    @property
    def x0(self) -> float:
        c = self.daf_composition
        return (c["H"] / M["H"]) / (c["C"] / M["C"])

    @property
    def y0(self) -> float:
        c = self.daf_composition
        return (c["O"] / M["O"]) / (c["C"] / M["C"])

    @property
    def z0(self) -> float:
        c = self.daf_composition
        return (c["N"] / M["N"]) / (c["C"] / M["C"])

    @property
    def molar_mass(self) -> float:
        return M["C"] + M["H"] * self.x0 + M["O"] * self.y0

    @property
    def cho_fraction(self) -> float:
        c = self.daf_composition
        return (c["C"] + c["H"] + c["O"]) / 100.0

    @property
    def fixed_carbon_daf(self) -> float:
        return self.fixed_carbon / self.daf_mass

    @property
    def ash_dry(self) -> float:
        return (self.ash / 100.0) / (1.0 - self.moisture / 100.0)

    @property
    def hhv(self) -> float:
        c = self.daf_composition
        scale = 100.0 / (c["C"] + c["H"] + c["O"])
        return channiwala(c["C"] * scale, c["H"] * scale, c["O"] * scale)

    @property
    def lhv(self) -> float:
        """Lower heating value on the reference basis, in MJ/kg.

        The latent heat of the water formed from the fuel hydrogen is
        subtracted so that the cold-gas efficiency compares like with like.
        """
        c = self.daf_composition
        scale = 100.0 / (c["C"] + c["H"] + c["O"])
        return self.hhv - VAPORISATION * WATER_PER_HYDROGEN * c["H"] * scale / 100.0


def channiwala(carbon: float, hydrogen: float, oxygen: float) -> float:
    """Higher heating value in MJ/kg from the CHO mass percentages."""
    return 0.3491 * carbon + 1.1783 * hydrogen - 0.1034 * oxygen


LAB_REPORT = "pellets_composition.xlsx"


def _lab_sheet(sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(DATA / LAB_REPORT, sheet_name=sheet)
    return frame.dropna(subset=["treatment"])


def load_feedstocks() -> dict[str, Feedstock]:
    """Characterisation of the three feedstocks from the laboratory report of the pellets.

    The report (Arroyo Dagobeth et al., 2025) gives the proximate analysis on the as-received basis, in five
    replicates, and the ultimate analysis on the dry basis, in three replicates, with the oxygen obtained by
    difference without subtracting the ash. The means of the replicates are converted here to the convention of
    Feedstock, in which carbon, hydrogen and nitrogen refer to the as-received sample and the hydrogen
    includes that of the moisture; the oxygen is then recomputed by difference including the ash. The blend was
    characterised as a material of its own.
    """
    proximate, ultimate = _lab_sheet("proximate"), _lab_sheet("ultimate")
    catalogue: dict[str, Feedstock] = {}
    for name in FEEDSTOCKS:
        code = GAS_LABEL[name]
        p = proximate[proximate["treatment"] == code]
        u = ultimate[ultimate["treatment"] == code]
        if p.empty or u.empty:
            raise ValueError(f"no characterisation of {code} in {LAB_REPORT}")
        moisture = float(p["MC"].mean())
        dry = 1.0 - moisture / 100.0
        catalogue[name] = Feedstock(
            label=name,
            moisture=moisture,
            ash=float(p["AC"].mean()),
            volatile_matter=float(p["VM"].mean()),
            fixed_carbon=float(p["FC"].mean()),
            carbon=float(u["C"].mean()) * dry,
            hydrogen=float(u["H"].mean()) * dry + moisture * 2.0 * M["H"] / MM["H2O"],
            nitrogen=float(u["N"].mean()) * dry,
        )
    return catalogue


def load_heating_values() -> pd.DataFrame:
    """Measured heating values of the pellets (bomb calorimeter, dry basis), means of three replicates."""
    frame = _lab_sheet("LHV")
    out = frame.groupby("treatment")[["HHV", "LHV"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out


def load_biochemical() -> pd.DataFrame:
    """Cellulose, hemicellulose, lignin and extractives of the pellets (mass fractions, dry basis), means."""
    frame = _lab_sheet("CHL")
    return frame.groupby("treatment")[["Cellulose", "Hemicellulose", "Lignin", "Extractives"]].mean()


def read_thermogram(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="latin1")
    frame.columns = ["T", "m"]
    frame = frame.dropna().sort_values("T").drop_duplicates("T")
    return frame["T"].to_numpy(float), frame["m"].to_numpy(float)


def thermogram_replicates(feedstock: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Individual runs of the 30 K/min campaign for one feedstock."""
    frame = pd.read_csv(DATA / THERMOGRAM_30K)
    subset = frame[frame["treatment"] == feedstock]
    if subset.empty:
        raise ValueError(f"no thermogram recorded for {feedstock}")
    runs = []
    for _, group in subset.groupby("replicate"):
        group = group.sort_values("T").drop_duplicates("T")
        runs.append((group["T"].to_numpy(float), group["m"].to_numpy(float)))
    return runs


def mean_thermogram(feedstock: str) -> tuple[tuple[np.ndarray, np.ndarray], list]:
    """Replicate-averaged thermogram on a common temperature grid.

    The runs are interpolated onto a shared grid before averaging, because the
    instrument samples on time and the recorded temperatures therefore differ
    slightly between runs.
    """
    runs = thermogram_replicates(feedstock)
    lower = max(float(t.min()) for t, _ in runs)
    upper = min(float(t.max()) for t, _ in runs)
    grid = np.linspace(lower, upper, 4000)
    stack = np.vstack([np.interp(grid, t, m) for t, m in runs])
    return (grid, stack.mean(axis=0)), runs


def organic_residue(
    temperature: np.ndarray | float,
    grid: np.ndarray,
    mass: np.ndarray,
    ash_fraction: float,
) -> np.ndarray:
    """Fraction of the organic matter remaining as solid, referred to T_REF."""
    mass_reference = np.interp(T_REF, grid, mass)
    ash_reference = ash_fraction / mass_reference
    residue = np.interp(temperature, grid, mass) / mass_reference
    return (residue - ash_reference) / (1.0 - ash_reference)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving quadratic regression, used only to locate the DTG extremum."""
    half = window // 2
    out = np.empty_like(values)
    index = np.arange(values.size)
    for i in index:
        lo, hi = max(0, i - half), min(values.size, i + half + 1)
        offset = index[lo:hi] - i
        out[i] = np.polyval(np.polyfit(offset, values[lo:hi], 2), 0.0)
    return out


@dataclass
class ThermalBasis:
    """Solid-yield trajectory and Fraser-Suzuki anchors for one treatment."""

    theta_nodes: np.ndarray
    theta_curve: Callable[[np.ndarray], np.ndarray]
    temperature_fine: np.ndarray
    residue_fine: np.ndarray
    rate_fine: np.ndarray
    peak_temperature: float
    onset_temperature: float
    floor_active: np.ndarray
    raw_nodes: np.ndarray
    theta_sd: np.ndarray | None = None
    replicates: int = 0

    @property
    def beta(self) -> float:
        return 1.0 / (self.peak_temperature - self.onset_temperature)


def build_thermal_basis(
    feedstock: Feedstock,
    thermogram: tuple[np.ndarray, np.ndarray],
    runs: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> ThermalBasis:
    """Build the solid-yield trajectory and apply the fixed-carbon floor.

    The floor guards against a thermogram that reports less solid than the fixed
    carbon of the same material. An open-pan thermobalance under flowing
    nitrogen sweeps the primary vapours away, whereas the volatile-matter
    crucible and the packed bed of the reactor both retain them in contact with
    the hot solid, so the thermobalance is the lower estimate of the two.
    """
    grid, mass = thermogram

    def raw_theta(temperature: np.ndarray | float) -> np.ndarray:
        return organic_residue(temperature, grid, mass, feedstock.ash / 100.0)

    floor = feedstock.fixed_carbon_daf

    def theta(temperature: np.ndarray | float) -> np.ndarray:
        return np.maximum(raw_theta(temperature), floor)

    upper = float(grid.max())
    fine = np.linspace(T_REF, upper, 900)
    residue = smooth(np.atleast_1d(raw_theta(fine)), 41)
    rate = -np.gradient(residue, fine)
    rate = smooth(rate, 31)

    peak = float(fine[int(np.argmax(rate))])
    below = np.flatnonzero(residue <= 0.98)
    onset = float(fine[below[0]]) if below.size else float(fine[0])

    theta_nodes = np.atleast_1d(theta(T_NODES))
    floor_active = np.atleast_1d(raw_theta(T_NODES)) < floor - 1e-12

    # Dispersion of the solid yield across the runs of the campaign, at each node
    # of the grid. Where a single run is available the dispersion is unknown and
    # is left to the caller.
    theta_sd, replicates = None, 0
    if runs:
        sample = np.vstack(
            [organic_residue(T_NODES, t, m, feedstock.ash / 100.0) for t, m in runs]
        )
        replicates = sample.shape[0]
        if replicates > 1:
            theta_sd = sample.std(axis=0, ddof=1)

    return ThermalBasis(
        raw_nodes=np.atleast_1d(raw_theta(T_NODES)),
        theta_nodes=theta_nodes,
        theta_curve=theta,
        temperature_fine=fine,
        residue_fine=np.maximum(residue, floor),
        rate_fine=rate,
        peak_temperature=peak,
        onset_temperature=onset,
        floor_active=floor_active,
        theta_sd=theta_sd,
        replicates=replicates,
    )


@dataclass
class GasData:
    mean: np.ndarray
    sd: np.ndarray
    replicates: np.ndarray


def injection_sequence() -> np.ndarray:
    """Released gas composition of the eight injections of one bag.

    The bag of the blend collected over the last stage, between 700 and 800
    degrees Celsius, was injected eight times in sequence. It is the only
    repeated measurement of the campaign, and it is returned in injection order.
    """
    frame = pd.read_excel(DATA / "gas_composition.xlsx")
    frame = frame.rename(columns={"T \u00b0C": "T"})
    case, temperature = REPEATED_INJECTIONS
    subset = _gas_subset(frame, case)
    values = subset[np.isclose(subset["T"], temperature)][
        ["vCO", "vCO2", "vCH4", "vH2"]
    ].to_numpy(float)
    return values / values.sum(axis=1, keepdims=True)


def load_gas_data() -> tuple[dict[tuple[str, str], GasData], np.ndarray]:
    """Released gas composition of the three pelletised feedstocks.

    Each cell carries one bag per stage, except the blend at 800 degrees
    Celsius, where one bag was injected eight times. No stage was sampled in
    replicate pyrolysis runs, so the run-to-run repeatability of the experiment
    is not measured. The relative dispersion of the eight injections is the only
    dispersion the campaign supplies, and it is assigned to every stage of every
    cell. It contains the repeatability of the chromatograph and the evolution
    of the bag over the injection sequence, but not the variability between
    pyrolysis runs, so it is a lower bound on the experimental dispersion; the
    article states it as an assumption, and the error budget of
    ccpyro.analyses doubles it to show what depends on it. The central value
    of the repeated bag is the mean of the three injections closest to the
    median composition, which discards the tails of the sequence without relying
    on a single reading.
    """
    frame = pd.read_excel(DATA / "gas_composition.xlsx")
    frame = frame.rename(columns={"T \u00b0C": "T"})
    columns = ["vCO", "vCO2", "vCH4", "vH2"]

    sequence = injection_sequence()
    relative_sd = sequence.std(axis=0, ddof=1) / sequence.mean(axis=0)

    catalogue: dict[tuple[str, str], GasData] = {}
    for case in CASES:
        subset = _gas_subset(frame, case)
        mean = np.empty((N_NODES, 4))
        counts = np.empty(N_NODES, dtype=int)
        for k, temperature in enumerate(T_NODES):
            group = subset[np.isclose(subset["T"], temperature)][columns].to_numpy(float)
            if group.size == 0:
                raise ValueError(f"missing chromatography for {case} at {temperature}")
            counts[k] = group.shape[0]
            if group.shape[0] > 1:
                centre = np.median(group, axis=0)
                order = np.argsort(np.linalg.norm(group - centre, axis=1))
                mean[k] = group[order[:3]].mean(axis=0)
            else:
                mean[k] = group[0]
        mean /= mean.sum(axis=1, keepdims=True)
        catalogue[case] = GasData(
            mean=mean,
            sd=np.maximum(relative_sd * mean, SIGMA["gc_floor"]),
            replicates=counts,
        )
    return catalogue, relative_sd


def _gas_subset(frame: pd.DataFrame, case: tuple[str, str]) -> pd.DataFrame:
    feedstock, densification = case
    return frame[
        (frame["Treatment"] == GAS_LABEL[feedstock])
        & (frame["Densification"] == GAS_DENSIFICATION[densification])
    ]


def tar_selectivity(
    temperature: np.ndarray | float,
    amplitude: float,
    alpha: float,
    beta: float,
    peak: float,
) -> np.ndarray:
    """Fraser-Suzuki peak with the continuous extension below its support."""
    argument = 1.0 + beta * (np.asarray(temperature, dtype=float) - peak)
    safe = np.maximum(argument, 1e-12)
    return np.where(argument > 1e-12, amplitude * np.exp(-alpha * np.log(safe) ** 2), 0.0)


def shape_from_parameters(alpha: float, beta: float, peak: float) -> dict[str, float]:
    s = np.sqrt(np.log(2.0) / alpha)
    rho = float(np.exp(s))
    return {
        "rho": rho,
        "T_L": peak - (1.0 - np.exp(-s)) / beta,
        "T_R": peak + (np.exp(s) - 1.0) / beta,
        "T_on": peak - 1.0 / beta,
    }


def oxygen_bound(
    composition: np.ndarray,
    theta: np.ndarray,
    y0: float,
    molar_mass: float,
    weighting: np.ndarray | None = None,
    water_minimum: float = 0.0,
    char_oxygen_minimum: float = 0.0,
    tar_oxygen_minimum: float = 0.0,
    tar_carbon_minimum: float = 0.10,
) -> pd.DataFrame:
    """Upper bound on the permanent-gas yield imposed by the feedstock oxygen.

    The bound constrains the gas accumulated up to each node, so it is evaluated
    with the composition of that accumulated gas. The chromatograph samples one
    stage at a time, so the stage compositions are combined using the organic
    mass released in each stage, which the thermogram supplies. The bound
    therefore remains independent of the estimation.
    """
    recorded = theta if weighting is None else weighting
    released = np.concatenate([[1.0 - recorded[0]], np.maximum(np.diff(-recorded), 0.0)])
    weights = np.cumsum(released[:, None] * composition, axis=0)
    accumulated = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    composition = accumulated

    carbon_fraction = composition[:, :3].sum(axis=1)
    demand = (composition[:, 0] + 2.0 * composition[:, 1]) / carbon_fraction
    available = np.maximum(
        0.0,
        y0
        - water_minimum
        - char_oxygen_minimum * theta * molar_mass / (M["C"] + M["H"] * 1.5 + M["O"] * 0.65)
        - tar_oxygen_minimum * tar_carbon_minimum,
    )
    carbon_gas = available / demand
    gas_by_oxygen = (carbon_gas / carbon_fraction) * (composition @ MM_GAS) / molar_mass
    gas_by_mass = np.maximum(0.0, 1.0 - theta - water_minimum * MM["H2O"] / molar_mass)
    gas_max = np.minimum(gas_by_oxygen, gas_by_mass)
    tar_min = np.maximum(0.0, gas_by_mass - gas_max)
    return pd.DataFrame(
        {
            "T": T_NODES,
            "lambda_O": demand,
            "w_gas_max_oxygen": gas_by_oxygen,
            "w_gas_max_mass": gas_by_mass,
            "w_gas_max": gas_max,
            "phi_tar_min": tar_min / np.maximum(tar_min + gas_max, 1e-12),
        }
    )


@dataclass
class ModelContext:
    """Everything the objective and the constraints need for one treatment."""

    feedstock: Feedstock
    thermal: ThermalBasis
    composition: np.ndarray
    composition_sd: np.ndarray
    alpha: float
    calibrate: bool = True
    differential: bool = True
    holdout: int = -1
    holdout_solid: int = -1
    relax_tar_monotonicity: bool = False
    anchor_scale: float = 1.0
    curvature_scale: float = 1.0
    c2_to_methane: np.ndarray | None = None
    condensate_end_anchors: bool = False

    x0: float = field(init=False)
    y0: float = field(init=False)
    molar_mass: float = field(init=False)
    theta: np.ndarray = field(init=False)
    theta_scale: np.ndarray = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.x0 = self.feedstock.x0
        self.y0 = self.feedstock.y0
        self.molar_mass = self.feedstock.molar_mass
        self.theta = self.thermal.theta_nodes
        # Scale of the solid-yield residual. The replicated campaign measures it
        # node by node, so a constant scale is used only where a feedstock
        # has no replicates.
        if self.thermal.theta_sd is not None:
            self.theta_scale = np.maximum(self.thermal.theta_sd, SIGMA["tga_floor"])
        else:
            self.theta_scale = np.full(N_NODES, SIGMA["tga"])
        upper = VAR_UPPER_TEMPLATE.copy()
        upper[IDX["x_c"]] = self.x0
        upper[IDX["y_c"]] = self.y0
        upper[IDX["x_t"]] = self.x0
        upper[IDX["y_t"]] = self.y0
        self.lower = np.tile(VAR_LOWER, N_NODES)
        self.upper = np.tile(upper.astype(float), N_NODES)
        if self.calibrate:
            self.lower = np.concatenate([self.lower, PARAM_LOWER])
            self.upper = np.concatenate([self.upper, PARAM_UPPER])

    @property
    def beta(self) -> float:
        return self.thermal.beta

    @property
    def peak(self) -> float:
        return self.thermal.peak_temperature


@dataclass
class Solution:
    """State of the model at every node of the temperature grid."""

    n_c: np.ndarray
    x_c: np.ndarray
    y_c: np.ndarray
    n_t: np.ndarray
    x_t: np.ndarray
    y_t: np.ndarray
    n_CO: np.ndarray
    n_CO2: np.ndarray
    n_CH4: np.ndarray
    n_H2: np.ndarray
    n_H2O: np.ndarray
    parameters: np.ndarray
    w_char: np.ndarray
    w_tar: np.ndarray
    w_water: np.ndarray
    w_gas: np.ndarray
    phi_model: np.ndarray
    phi_tar: np.ndarray
    objective: float
    feasibility: float
    n_C2: np.ndarray | None = None

    @property
    def w_liquid(self) -> np.ndarray:
        return self.w_tar + self.w_water

    @property
    def gas_moles(self) -> np.ndarray:
        total = self.n_CO + self.n_CO2 + self.n_CH4 + self.n_H2
        return total if self.n_C2 is None else total + self.n_C2


def unpack(vector: np.ndarray, context: ModelContext) -> Solution:
    block = vector[: N_NODES * N_FREE].reshape(N_NODES, N_FREE)
    if context.calibrate:
        parameters = vector[N_NODES * N_FREE :]
    else:
        parameters = np.array([KAPPA_PRIOR, AMPLITUDE_PRIOR, PRIOR_MEAN[2], PRIOR_MEAN[3]])

    n_c, x_c, y_c, x_t, y_t, n_CO, n_CO2, n_CH4 = (block[:, IDX[name]] for name in VARIABLES)

    # Hydrocarbons of two and three carbon atoms, absent from the chromatographic
    # method, enter the carbon and hydrogen balances when the omitted-species
    # budget is evaluated. The lump is tied to the released methane by a
    # literature ratio, so it introduces no free variable and no circularity.
    if context.c2_to_methane is None:
        n_C2 = None
        carbon_C2 = 0.0
        hydrogen_C2 = 0.0
        mass_C2 = 0.0
    else:
        released_CH4 = np.concatenate([[n_CH4[0]], np.diff(n_CH4)])
        n_C2 = np.cumsum(context.c2_to_methane * np.maximum(released_CH4, 0.0))
        carbon_C2 = C2_CARBON * n_C2
        hydrogen_C2 = C2_HYDROGEN * n_C2
        mass_C2 = MM_C2 * n_C2

    n_t = 1.0 - n_c - n_CO - n_CO2 - n_CH4 - carbon_C2
    n_H2O = context.y0 - y_c * n_c - y_t * n_t - n_CO - 2.0 * n_CO2
    n_H2 = 0.5 * (
        context.x0 - x_c * n_c - x_t * n_t - 2.0 * n_H2O - 4.0 * n_CH4 - hydrogen_C2
    )

    molar_mass_char = M["C"] + M["H"] * x_c + M["O"] * y_c
    molar_mass_tar = M["C"] + M["H"] * x_t + M["O"] * y_t
    gas = np.column_stack([n_CO, n_CO2, n_CH4, n_H2])
    mass_gas = gas @ MM_GAS + mass_C2
    mass_tar = n_t * molar_mass_tar

    released = (
        np.vstack([gas[0], np.diff(gas, axis=0)]) if context.differential else gas
    )
    positive = np.maximum(released, 0.0)
    phi_model = positive / np.maximum(positive.sum(axis=1, keepdims=True), 1e-9)

    return Solution(
        n_c=n_c,
        x_c=x_c,
        y_c=y_c,
        n_t=n_t,
        x_t=x_t,
        y_t=y_t,
        n_CO=n_CO,
        n_CO2=n_CO2,
        n_CH4=n_CH4,
        n_H2=n_H2,
        n_H2O=n_H2O,
        parameters=parameters,
        w_char=n_c * molar_mass_char / context.molar_mass,
        w_tar=mass_tar / context.molar_mass,
        w_water=n_H2O * MM["H2O"] / context.molar_mass,
        w_gas=mass_gas / context.molar_mass,
        phi_model=phi_model,
        phi_tar=mass_tar / np.maximum(mass_tar + mass_gas, 1e-12),
        objective=np.nan,
        feasibility=np.nan,
        n_C2=n_C2,
    )


def objective_terms(vector: np.ndarray, context: ModelContext) -> dict[str, float]:
    """Decompose the objective into a measurement part and a structural part.

    The identifiability analysis needs the two separated. Only the solid-yield and
    gas-composition blocks are measurements of this experiment; the closures, the
    anchors, the curvature penalty and the parameter prior are structural
    knowledge imported into the estimate, and the curvature they supply is not
    information the measurements supply.
    """
    state = unpack(vector, context)
    kappa, amplitude, a_H, a_O = state.parameters

    ratio = kappa * MM["CO2"] / MM["H2O"]
    water_target = ratio / (1.0 + ratio)
    water_model = state.n_H2O / np.maximum(state.n_H2O + state.n_CO2, 1e-12)
    tar_target = tar_selectivity(
        T_NODES, amplitude, context.alpha, context.beta, context.peak
    )

    solid_residual = (state.w_char - context.theta) / context.theta_scale
    if 0 <= context.holdout_solid < N_NODES:
        solid_residual = np.delete(solid_residual, context.holdout_solid)
    gas_residual = (state.phi_model - context.composition) / context.composition_sd
    if 0 <= context.holdout < N_NODES:
        gas_residual = np.delete(gas_residual, context.holdout, axis=0)

    scale_h = SIGMA["van_krevelen"] * context.x0
    scale_o = SIGMA["van_krevelen"] * context.y0

    # The two anchors on the end-of-range composition of the condensable organics
    # are not applied in the reported case. No elemental analysis of a pyrolysis
    # condensate above 600 degrees Celsius could be verified in the literature, so
    # a target at 800 degrees Celsius cannot be justified independently, and an
    # anchor that is not justified determines an endpoint that is then reported as
    # a finding. They are retained here only for the structural sensitivity.
    anchors = [
        (state.x_t[0] - context.x0) / ANCHORS["x_t_first_sd"],
        (state.y_t[0] - context.y0) / ANCHORS["y_t_first_sd"],
        (state.x_c[-1] - ANCHORS["x_c_last"]) / ANCHORS["x_c_last_sd"],
        (state.y_c[-1] - ANCHORS["y_c_last"]) / ANCHORS["y_c_last_sd"],
    ]
    if context.condensate_end_anchors:
        anchors += [
            (state.x_t[-1] - ANCHORS["x_t_last"]) / ANCHORS["x_t_last_sd"],
            (state.y_t[-1] - ANCHORS["y_t_last"]) / ANCHORS["y_t_last_sd"],
        ]
    anchor_residuals = np.array(anchors)

    penalty = context.curvature_scale * OMEGA["regularization"] / SIGMA["regularization"] ** 2
    curvature = 0.0
    for values, scale in (
        (state.x_c, context.x0),
        (state.y_c, context.y0),
        (state.x_t, context.x0),
        (state.y_t, context.y0),
    ):
        trajectory = np.concatenate([[1.0], values / scale])
        curvature += penalty * np.sum(np.diff(trajectory, n=2) ** 2)

    return {
        "solid": OMEGA["tga"] * np.sum(solid_residual**2),
        "gas": OMEGA["gc"] * np.sum(gas_residual**2),
        "water_closure": OMEGA["water"]
        * np.sum(((water_model - water_target) / SIGMA["water"]) ** 2),
        "tar_closure": OMEGA["tar"]
        * np.sum(((state.phi_tar - tar_target) / SIGMA["tar"]) ** 2),
        "carbonisation": OMEGA["van_krevelen"]
        * (
            np.sum(((state.x_c - context.x0 * context.theta**a_H) / scale_h) ** 2)
            + np.sum(((state.y_c - context.y0 * context.theta**a_O) / scale_o) ** 2)
        ),
        "anchors": np.sum((anchor_residuals / context.anchor_scale) ** 2),
        "curvature": curvature,
        "prior": (
            OMEGA["prior"] * np.sum(((state.parameters - PRIOR_MEAN) / PRIOR_SD) ** 2)
            if context.calibrate
            else 0.0
        ),
    }


# The three nested levels of evidence. The first contains only what this
# experiment measured; the second adds the empirical correlations transferred
# from other work; the third adds the anchors, the smoothness penalty and the
# parameter prior, and is what the estimator minimises. Reporting the three
# separately shows which statement rests on which kind of knowledge.
DATA_BLOCKS = ("solid", "gas")
CLOSURE_BLOCKS = DATA_BLOCKS + ("water_closure", "tar_closure", "carbonisation")
LEVELS = {
    "measurements": DATA_BLOCKS,
    "closures": CLOSURE_BLOCKS,
    "estimator": None,
}


def level_misfit(vector: np.ndarray, context: ModelContext, level: str) -> float:
    """Value of the criterion associated with one level of evidence."""
    terms = objective_terms(vector, context)
    blocks = LEVELS[level]
    if blocks is None:
        return float(sum(terms.values()))
    return float(sum(terms[key] for key in blocks))


def objective(vector: np.ndarray, context: ModelContext) -> float:
    return sum(objective_terms(vector, context).values()) / OBJECTIVE_SCALE


def data_misfit(vector: np.ndarray, context: ModelContext) -> float:
    """Sum of squared standardised residuals of the measurements alone.

    This is the statistic on which the profile-likelihood thresholds are
    evaluated, because it is the only part of the objective to which a
    measurement of this experiment contributes.
    """
    return level_misfit(vector, context, "measurements")


def data_objective(vector: np.ndarray, context: ModelContext) -> float:
    """Numerically scaled data misfit, for use as an optimisation objective."""
    return data_misfit(vector, context) / OBJECTIVE_SCALE


def closure_misfit(vector: np.ndarray, context: ModelContext) -> float:
    """Measurements together with the transferred empirical correlations."""
    return level_misfit(vector, context, "closures")


def closure_objective(vector: np.ndarray, context: ModelContext) -> float:
    return closure_misfit(vector, context) / OBJECTIVE_SCALE


def constraints(vector: np.ndarray, context: ModelContext) -> np.ndarray:
    """Inequality vector, admissible when every component is non-negative."""
    state = unpack(vector, context)
    increments = [
        np.concatenate([[value[0]], np.diff(value)])
        for value in (state.n_CO, state.n_CO2, state.n_CH4, state.n_H2, state.n_H2O)
    ]
    return np.concatenate(
        [
            state.n_t,
            state.n_H2O,
            state.n_H2,
            -np.diff(np.concatenate([[1.0], state.n_c])),
            -np.diff(np.concatenate([[context.x0], state.x_c])),
            -np.diff(np.concatenate([[context.y0], state.y_c])),
            *(
                []
                if context.relax_tar_monotonicity
                else [-np.diff(state.x_t), -np.diff(state.y_t)]
            ),
            state.x_t - state.x_c,
            state.y_t - state.y_c,
            *increments,
        ]
    )


def initial_point(context: ModelContext, a_H: float = 1.05, a_O: float = 1.30) -> np.ndarray:
    """Feasible start built from the single-index carbonisation closure."""
    theta = context.theta
    x_c = context.x0 * theta**a_H
    y_c = context.y0 * theta**a_O
    n_c = theta * context.molar_mass / (M["C"] + M["H"] * x_c + M["O"] * y_c)

    ramp = np.linspace(0.0, 1.0, N_NODES)
    x_t = np.minimum.accumulate(
        np.minimum(context.x0, np.maximum(x_c * 1.03, context.x0 - (context.x0 - ANCHORS["x_t_last"]) * ramp))
    )
    y_t = np.minimum.accumulate(
        np.minimum(context.y0, np.maximum(y_c * 1.03, context.y0 - (context.y0 - ANCHORS["y_t_last"]) * ramp))
    )

    block = np.zeros((N_NODES, N_FREE))
    block[:, IDX["n_c"]] = n_c
    block[:, IDX["x_c"]] = x_c
    block[:, IDX["y_c"]] = y_c
    block[:, IDX["x_t"]] = x_t
    block[:, IDX["y_t"]] = y_t

    volatile_carbon = 1.0 - n_c
    fractions = context.composition[:, :3] / context.composition[:, :3].sum(axis=1, keepdims=True)
    for k in range(N_NODES):
        best, best_penalty = np.array([1e-4, 1e-4, 1e-5]), np.inf
        for share in np.linspace(0.001, 0.98, 400):
            gas = share * volatile_carbon[k] * fractions[k]
            n_t = volatile_carbon[k] * (1.0 - share)
            n_H2O = context.y0 - y_c[k] * n_c[k] - y_t[k] * n_t - gas[0] - 2.0 * gas[1]
            n_H2 = 0.5 * (
                context.x0 - x_c[k] * n_c[k] - x_t[k] * n_t - 2.0 * n_H2O - 4.0 * gas[2]
            )
            penalty = np.sum(np.minimum(0.0, [n_H2O, n_H2, n_t]) ** 2) - 1e-6 * share
            if penalty < best_penalty:
                best_penalty, best = penalty, gas
        block[k, IDX["n_CO"] : IDX["n_CH4"] + 1] = best
    for name in ("n_CO", "n_CO2", "n_CH4"):
        block[:, IDX[name]] = np.maximum.accumulate(block[:, IDX[name]])

    vector = block.reshape(-1)
    if context.calibrate:
        vector = np.concatenate([vector, PRIOR_MEAN])
    return np.clip(vector, context.lower + 1e-9, context.upper - 1e-9)


def restore_feasibility(
    vector: np.ndarray,
    context: ModelContext,
    margin: float = 1e-4,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Project an inadmissible start onto the feasible set before optimisation."""

    def violation(candidate: np.ndarray) -> float:
        return float(np.sum(np.minimum(0.0, constraints(candidate, context) - margin) ** 2))

    if np.min(constraints(vector, context)) > -tolerance:
        return vector
    result = minimize(
        violation,
        vector,
        method="L-BFGS-B",
        bounds=list(zip(context.lower, context.upper)),
        options={"maxiter": 300, "ftol": 1e-12},
    )
    return result.x if violation(result.x) < violation(vector) else vector


def solve(
    context: ModelContext,
    start: np.ndarray | Sequence[np.ndarray] | None = None,
    restarts: int = 3,
    max_iterations: int = 700,
    tolerance: float = 1e-12,
) -> Solution:
    """Sequential quadratic programming from one or several admissible starts."""
    if start is None:
        candidates = [initial_point(context)]
    elif isinstance(start, np.ndarray):
        candidates = [start]
    else:
        candidates = list(start)

    if len(candidates) > 2:
        merit = [
            objective(np.clip(c, context.lower, context.upper), context)
            + 1e4 * np.sum(np.minimum(0.0, constraints(np.clip(c, context.lower, context.upper), context)) ** 2)
            for c in candidates
        ]
        candidates = [candidates[i] for i in np.argsort(merit)[:2]]

    best_vector, best_value = None, np.inf
    for candidate in candidates:
        vector = restore_feasibility(
            np.clip(candidate, context.lower + 1e-9, context.upper - 1e-9), context
        )
        value = objective(vector, context)
        if value < best_value:
            best_vector, best_value = vector, value

        previous = np.inf
        for _ in range(restarts):
            result = minimize(
                objective,
                vector,
                args=(context,),
                method="SLSQP",
                bounds=list(zip(context.lower, context.upper)),
                constraints=[{"type": "ineq", "fun": constraints, "args": (context,)}],
                options={"maxiter": max_iterations, "ftol": tolerance},
            )
            vector = result.x
            admissible = np.min(constraints(vector, context)) > -1e-6
            if admissible and result.fun < best_value:
                best_vector, best_value = result.x.copy(), result.fun
            if previous - result.fun < 1e-10 * max(1.0, abs(result.fun)):
                break
            previous = result.fun

    state = unpack(best_vector, context)
    state.objective = best_value
    state.feasibility = float(np.min(constraints(best_vector, context)))
    state.vector = best_vector
    return state


def performance_indices(state: Solution, context: ModelContext) -> dict[str, np.ndarray]:
    """Heating value of the released gas, cumulative volume, hydrogen yield and CGE.

    The heating value refers to the gas released over each stage, which is what
    the chromatograph samples. The energy content and the cold-gas efficiency
    refer to the gas accumulated up to the node, so both are evaluated with the
    accumulated composition rather than with the released one.
    """
    calorific = np.array([LHV_VOLUMETRIC[s] for s in GAS_SPECIES])
    accumulated = np.column_stack([state.n_CO, state.n_CO2, state.n_CH4, state.n_H2])
    released = np.vstack([accumulated[0], np.diff(accumulated, axis=0)])
    energy_released = np.maximum(released, 0.0) @ calorific
    moles_released = np.maximum(released, 0.0).sum(axis=1)
    energy_accumulated = accumulated @ calorific

    if state.n_C2 is not None:
        released_C2 = np.maximum(
            np.concatenate([[state.n_C2[0]], np.diff(state.n_C2)]), 0.0
        )
        energy_released = energy_released + released_C2 * LHV_C2
        moles_released = moles_released + released_C2
        energy_accumulated = energy_accumulated + state.n_C2 * LHV_C2

    heating_value = energy_released / np.maximum(moles_released, 1e-12)

    moles_per_kg_daf = 1000.0 / context.molar_mass
    gas_volume = state.gas_moles * moles_per_kg_daf * MOLAR_VOLUME
    hydrogen_yield = state.n_H2 * moles_per_kg_daf * MOLAR_VOLUME * 1000.0

    gas_energy = energy_accumulated * moles_per_kg_daf * MOLAR_VOLUME
    efficiency = 100.0 * gas_energy / context.feedstock.lhv

    char_hhv = np.array(
        [
            channiwala(*mass_percentages(1.0, xc, yc))
            for xc, yc in zip(state.x_c, state.y_c)
        ]
    )
    tar_hhv = np.array(
        [
            channiwala(*mass_percentages(1.0, xt, yt))
            for xt, yt in zip(state.x_t, state.y_t)
        ]
    )
    return {
        "LHV_gas": heating_value,
        "V_gas": gas_volume,
        "Y_H2": hydrogen_yield,
        "E_gas": gas_energy,
        "CGE": efficiency,
        "HHV_char": char_hhv,
        "HHV_tar": tar_hhv,
    }


def mass_percentages(carbon: float, x: float, y: float) -> tuple[float, float, float]:
    molar_mass = M["C"] * carbon + M["H"] * x + M["O"] * y
    return (
        100.0 * M["C"] * carbon / molar_mass,
        100.0 * M["H"] * x / molar_mass,
        100.0 * M["O"] * y / molar_mass,
    )


def dry_basis(state: Solution, context: ModelContext) -> dict[str, np.ndarray]:
    ash = context.feedstock.ash_dry
    return {
        "char": state.w_char * (1.0 - ash) + ash,
        "liquid": state.w_liquid * (1.0 - ash),
        "gas": state.w_gas * (1.0 - ash),
    }


TRACKED = (
    "w_char",
    "w_tar",
    "w_water",
    "w_gas",
    "w_liquid",
    "LHV_gas",
    "V_gas",
    "Y_H2",
    "CGE",
    "x_c",
    "y_c",
    "x_t",
    "y_t",
    "water_in_liquid",
    "HHV_char",
)


def collect(state: Solution, context: ModelContext) -> dict[str, np.ndarray]:
    indices = performance_indices(state, context)
    record = {
        "w_char": state.w_char,
        "w_tar": state.w_tar,
        "w_water": state.w_water,
        "w_gas": state.w_gas,
        "w_liquid": state.w_liquid,
        "x_c": state.x_c,
        "y_c": state.y_c,
        "x_t": state.x_t,
        "y_t": state.y_t,
        "water_in_liquid": state.w_water / np.maximum(state.w_liquid, 1e-12),
    }
    record.update({key: indices[key] for key in ("LHV_gas", "V_gas", "Y_H2", "CGE", "HHV_char")})
    return record


def perturb_characterisation(
    feedstock: Feedstock, thermal: ThermalBasis, rng: np.random.Generator
) -> tuple[Feedstock, ThermalBasis]:
    """Draw one systematic offset per treatment for the characterisation.

    Analytical error in the proximate and ultimate analyses is a property of the
    determination and not of the temperature node, so a single offset is drawn
    for each realisation and applied to the whole trajectory. The same holds for
    the thermogravimetric mass scale.
    """
    shifted = replace(
        feedstock,
        carbon=feedstock.carbon + rng.normal(0.0, SIGMA_ANALYSIS["carbon"]),
        hydrogen=max(0.1, feedstock.hydrogen + rng.normal(0.0, SIGMA_ANALYSIS["hydrogen"])),
        nitrogen=max(0.0, feedstock.nitrogen + rng.normal(0.0, SIGMA_ANALYSIS["nitrogen"])),
        ash=max(0.0, feedstock.ash + rng.normal(0.0, SIGMA_ANALYSIS["ash"])),
        moisture=max(0.0, feedstock.moisture + rng.normal(0.0, SIGMA_ANALYSIS["moisture"])),
        fixed_carbon=max(0.0, feedstock.fixed_carbon + rng.normal(0.0, SIGMA_ANALYSIS["fixed_carbon"])),
    )
    # Run-to-run variation of the thermogram is measured by the replicates of the
    # campaign and is coherent along the whole curve, since it originates in the
    # loading of the pan and in the mass calibration rather than at a temperature
    # node. One standard normal is therefore drawn per realisation and applied to
    # the measured dispersion of every node. Where a feedstock has no replicates
    # a relative mass-scale assumption is used instead.
    if thermal.theta_sd is not None:
        nodes = thermal.raw_nodes + rng.normal() * thermal.theta_sd
    else:
        nodes = thermal.raw_nodes * (1.0 + rng.normal(0.0, SIGMA_THETA_RELATIVE))
    nodes = np.maximum(nodes, shifted.fixed_carbon_daf)
    nodes = np.minimum.accumulate(np.clip(nodes, 1e-3, 1.0))
    return shifted, replace(thermal, theta_nodes=nodes)


def perturb_composition(
    base: np.ndarray,
    relative_sd: np.ndarray,
    rng: np.random.Generator,
    model: str = "by_difference",
) -> np.ndarray:
    """Draw one realisation of the four-species gas composition.

    Carbon dioxide is not measured. It is obtained by difference over the six
    integrated channels, and after the oxygen and nitrogen of the admitted air
    are removed the by-difference construction survives as the identity
    phi_CO2 = 1 - phi_CO - phi_CH4 - phi_H2. Its error is therefore fully
    determined by the errors of the three measured channels and has no
    independent dispersion of its own. The default model reproduces that
    structure: the three measured channels are perturbed with their replicate
    dispersion, carbon dioxide follows by difference, and an independent term
    carries what does not cancel on renormalisation, namely the argon of the
    entrained air and whatever the oxygen channel fails to record. The
    alternative model perturbs all four channels independently and is retained
    only to quantify what that simplification costs.
    """
    if model == "independent":
        draw = base * (1.0 + rng.normal(0.0, relative_sd, size=base.shape))
        draw = np.clip(draw, 1e-6, None)
        return draw / draw.sum(axis=1, keepdims=True)

    if model != "by_difference":
        raise ValueError(f"unknown gas error model {model!r}")

    measured = [0, 2, 3]  # CO, CH4, H2
    draw = base.copy()
    for index in measured:
        draw[:, index] = base[:, index] * (
            1.0 + rng.normal(0.0, relative_sd[index], size=base.shape[0])
        )
    draw[:, 1] = (
        1.0
        - draw[:, measured].sum(axis=1)
        + rng.normal(0.0, SIGMA_AIR_RESIDUAL, size=base.shape[0])
    )
    draw = np.clip(draw, 1e-6, None)
    return draw / draw.sum(axis=1, keepdims=True)


def propagate(
    context: ModelContext,
    reference: Solution,
    relative_sd: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    characterisation: bool = True,
    gas_error_model: str = "by_difference",
) -> dict[str, np.ndarray]:
    """Resample the measurements and re-solve from the reference state."""
    samples: dict[str, list[np.ndarray]] = {key: [] for key in TRACKED}
    base = context.composition
    start = reference.vector

    for _ in range(draws):
        perturbed = perturb_composition(base, relative_sd, rng, gas_error_model)
        feedstock, thermal = (
            perturb_characterisation(context.feedstock, context.thermal, rng)
            if characterisation
            else (context.feedstock, context.thermal)
        )
        trial = ModelContext(
            feedstock=feedstock,
            thermal=thermal,
            composition=perturbed,
            composition_sd=context.composition_sd,
            alpha=context.alpha,
            calibrate=context.calibrate,
            differential=context.differential,
        )
        state = solve(trial, start=start, restarts=1, max_iterations=60, tolerance=1e-9)
        if state.feasibility < -1e-4:
            continue
        record = collect(state, trial)
        for key in TRACKED:
            samples[key].append(record[key])

    return {key: np.vstack(values) for key, values in samples.items() if values}


def summarise(samples: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for k, temperature in enumerate(T_NODES):
        row = {"T": temperature}
        for key in TRACKED:
            row[key] = reference[key][k]
            row[f"{key}_sd"] = samples[key][:, k].std(ddof=1) if key in samples else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hessian_spectrum(context: ModelContext, reference: Solution) -> pd.DataFrame:
    """Curvature of the objective at the optimum, direction by direction.

    Directions whose curvature is far below the largest are not resolved by the
    measurements and are held by the priors, the anchors and the curvature
    penalty rather than by the data.
    """
    vector = np.asarray(reference.vector, float)
    span = np.maximum(context.upper - context.lower, 1e-6)
    step = 1e-4 * span
    size = vector.size
    matrix = np.zeros((size, size))
    base = objective(vector, context)
    for i in range(size):
        for j in range(i, size):
            shift_i, shift_j = np.zeros(size), np.zeros(size)
            shift_i[i], shift_j[j] = step[i], step[j]
            plus = objective(vector + shift_i + shift_j, context)
            minus = objective(vector - shift_i - shift_j, context)
            cross = objective(vector + shift_i - shift_j, context)
            anti = objective(vector - shift_i + shift_j, context)
            matrix[i, j] = matrix[j, i] = (plus + minus - cross - anti) / (4.0 * step[i] * step[j])
    scaled = matrix * np.outer(span, span)
    values = np.linalg.eigvalsh(0.5 * (scaled + scaled.T))
    values = np.sort(values)[::-1]
    largest = values[0]
    return pd.DataFrame(
        {
            "index": np.arange(1, size + 1),
            "eigenvalue": values,
            "relative": values / largest,
            "objective_at_optimum": base,
        }
    )


def holdout_validation(context: ModelContext, reference: Solution) -> pd.DataFrame:
    """Withhold one measurement at a time and predict it from the rest.

    Two tests are run. Removing the solid-yield residual at one node asks
    whether the gas composition and the balances alone recover the
    thermogravimetric yield there, which is a prediction the framework claims to
    support. Removing the chromatographic residual instead asks whether the gas
    composition at a node is determined by anything other than its own
    measurement, which is a question about identifiability.
    """
    rows = []
    for node in range(N_NODES):
        solid = solve(
            replace(context, holdout_solid=node), start=reference.vector, restarts=1
        )
        rows.append(
            {
                "test": "solid yield",
                "T": T_NODES[node],
                "quantity": "theta",
                "measured": context.theta[node],
                "predicted": solid.w_char[node],
                "fitted": reference.w_char[node],
            }
        )
        gas = solve(
            replace(context, holdout=node), start=reference.vector, restarts=1
        )
        for index, species in enumerate(GAS_SPECIES):
            rows.append(
                {
                    "test": "gas composition",
                    "T": T_NODES[node],
                    "quantity": species,
                    "measured": context.composition[node, index],
                    "predicted": gas.phi_model[node, index],
                    "fitted": reference.phi_model[node, index],
                }
            )
    frame = pd.DataFrame(rows)
    frame["error"] = frame["predicted"] - frame["measured"]
    return frame


def structural_sensitivity(context: ModelContext, reference: Solution) -> pd.DataFrame:
    """Response of the estimate to the choices that are not measurements."""
    start = reference.vector
    cases = {
        "reference": {},
        "anchors halved": {"anchor_scale": 2.0},
        "anchors quartered": {"anchor_scale": 4.0},
        "no curvature penalty": {"curvature_scale": 0.0},
        "curvature penalty quadrupled": {"curvature_scale": 4.0},
        "tar monotonicity relaxed": {"relax_tar_monotonicity": True},
    }
    rows = []
    for name, overrides in cases.items():
        trial = replace(context, **overrides)
        state = solve(trial, start=start, restarts=1)
        indices = performance_indices(state, trial)
        rows.append(
            {
                "case": name,
                "objective": state.objective,
                "w_liquid_500": state.w_liquid[2],
                "w_char_800": state.w_char[-1],
                "x_t_800": state.x_t[-1],
                "y_t_800": state.y_t[-1],
                "y_t_400": state.y_t[1],
                "CGE_800": indices["CGE"][-1],
            }
        )
    for scale in (0.95, 1.05):
        nodes = np.minimum.accumulate(
            np.clip(np.maximum(context.thermal.raw_nodes * scale, context.feedstock.fixed_carbon_daf), 1e-3, 1.0)
        )
        trial = replace(context, thermal=replace(context.thermal, theta_nodes=nodes))
        state = solve(trial, start=start, restarts=1)
        indices = performance_indices(state, trial)
        rows.append(
            {
                "case": f"solid yield scaled by {scale:.2f}",
                "objective": state.objective,
                "w_liquid_500": state.w_liquid[2],
                "w_char_800": state.w_char[-1],
                "x_t_800": state.x_t[-1],
                "y_t_800": state.y_t[-1],
                "y_t_400": state.y_t[1],
                "CGE_800": indices["CGE"][-1],
            }
        )
    return pd.DataFrame(rows)


def sensitivity_to_alpha(context: ModelContext, reference: Solution) -> pd.DataFrame:
    rows = []
    for alpha in ALPHA_SENSITIVITY:
        trial = ModelContext(
            feedstock=context.feedstock,
            thermal=context.thermal,
            composition=context.composition,
            composition_sd=context.composition_sd,
            alpha=alpha,
            calibrate=context.calibrate,
            differential=context.differential,
        )
        state = solve(trial, start=reference.vector, restarts=2)
        rows.append(
            {
                "alpha": alpha,
                "objective": state.objective,
                "A": state.parameters[1],
                "w_liquid_500": state.w_liquid[2],
                "w_gas_500": state.w_gas[2],
                "w_liquid_800": state.w_liquid[-1],
                "w_gas_800": state.w_gas[-1],
                "x_c_800": state.x_c[-1],
            }
        )
    return pd.DataFrame(rows)


def verification(state: Solution, context: ModelContext) -> dict[str, float]:
    carbon = state.n_c + state.n_t + state.n_CO + state.n_CO2 + state.n_CH4 - 1.0
    hydrogen = (
        state.x_c * state.n_c
        + state.x_t * state.n_t
        + 2.0 * state.n_H2O
        + 4.0 * state.n_CH4
        + 2.0 * state.n_H2
        - context.x0
    )
    oxygen = (
        state.y_c * state.n_c
        + state.y_t * state.n_t
        + state.n_H2O
        + state.n_CO
        + 2.0 * state.n_CO2
        - context.y0
    )
    closure = state.w_char + state.w_tar + state.w_water + state.w_gas - 1.0
    return {
        "carbon": float(np.max(np.abs(carbon))),
        "hydrogen": float(np.max(np.abs(hydrogen))),
        "oxygen": float(np.max(np.abs(oxygen))),
        "mass_closure": float(np.max(np.abs(closure))),
        "feasibility": state.feasibility,
        "objective": state.objective,
    }


SHORT = {
    "rice husk (pelletized)": "huskp",
    "blend (pelletized)": "blendp",
    "corncob (pelletized)": "cobp",
}


def write_figure_data() -> None:
    """Reshape the result tables into the wide format consumed by pgfplots."""
    FIGDATA.mkdir(exist_ok=True)

    yields = pd.read_csv(RESULTS / "product_yields.csv")
    columns = [
        "w_char", "w_tar", "w_water", "w_gas", "w_liquid", "x_c", "y_c", "x_t",
        "y_t", "water_in_liquid", "LHV_gas", "V_gas", "Y_H2", "CGE", "HHV_char",
        "char_dry", "liquid_dry", "gas_dry", "phi_tar_model",
    ]
    columns += [f"{name}_sd" for name in
                ("w_char", "w_tar", "w_water", "w_gas", "w_liquid", "LHV_gas",
                 "Y_H2", "CGE", "x_c", "y_c", "x_t", "y_t", "water_in_liquid")]
    columns += [f"phi_{species}_{kind}" for species in GAS_SPECIES
                for kind in ("model", "exp", "sd")]

    wide = pd.DataFrame({"T": T_NODES})
    for label, code in SHORT.items():
        subset = yields[yields["treatment"] == label].sort_values("T")
        for name in columns:
            if name in subset:
                wide[f"{name}_{code}"] = subset[name].to_numpy()
    wide.to_csv(FIGDATA / "yields.csv", index=False)

    curves = pd.read_csv(FIGDATA / "thermal_curves.csv")
    for label, code in SHORT.items():
        subset = curves[curves["treatment"] == label]
        subset[["T", "theta", "rate"]].to_csv(FIGDATA / f"thermal_{code}.csv", index=False)

    frame = pd.read_csv(RESULTS / "oxygen_bound.csv")
    bound = pd.DataFrame({"T": T_NODES})
    for label, code in SHORT.items():
        subset = frame[frame["treatment"] == label].sort_values("T")
        for name in ("w_gas_max", "phi_tar_min", "phi_tar_literature"):
            bound[f"{name}_{code}"] = subset[name].to_numpy()
    bound.to_csv(FIGDATA / "oxygen_bound.csv", index=False)

    calibration = pd.read_csv(RESULTS / "model_calibration.csv")
    grid = np.linspace(220.0, 900.0, 341)
    shape = pd.DataFrame({"T": grid})
    for _, row in calibration.iterrows():
        shape[f"phi_{SHORT[row['treatment']]}"] = tar_selectivity(
            grid, row["A"], row["alpha"], row["beta"], row["T_M"]
        )
    shape["phi_literature"] = tar_selectivity(grid, AMPLITUDE_PRIOR, ALPHA_TRANSFERRED, 0.008, 338.16)
    shape.to_csv(FIGDATA / "tar_shape.csv", index=False)


BEST_VECTORS = RESULTS / "best_vectors.npz" if (RESULTS / "best_vectors.npz").exists() else PACKAGE_DATA / "best_vectors.npz"


def cached_starts(label: str) -> list:
    """Best decision vector found for a cell in any previous search."""
    if not BEST_VECTORS.exists():
        return []
    with np.load(BEST_VECTORS) as archive:
        return [archive[label]] if label in archive.files else []


def store_best(states: dict) -> None:
    """Record the best vector of every cell, merging with what is already there."""
    RESULTS.mkdir(exist_ok=True)
    kept = {}
    if BEST_VECTORS.exists():
        with np.load(BEST_VECTORS) as archive:
            kept = {name: archive[name] for name in archive.files}
    kept.update({label: state.vector for label, state in states.items()})
    np.savez(RESULTS / "best_vectors.npz", **kept)   # never into the installed package


def state_from_vector(vector: np.ndarray, context: ModelContext) -> Solution | None:
    """Solution for a stored decision vector, or None if it is not admissible."""
    vector = np.asarray(vector, float)
    if vector.shape != context.lower.shape:
        return None
    feasibility = float(np.min(constraints(vector, context)))
    if feasibility < -1e-6 or np.any(vector < context.lower - 1e-9) or np.any(vector > context.upper + 1e-9):
        return None
    state = unpack(vector, context)
    state.objective = objective(vector, context)
    state.feasibility = feasibility
    state.vector = vector.copy()
    return state


def build_cases(passes: int = 2, random_starts: int = RANDOM_STARTS) -> dict:
    """Assemble and solve every cell of the design.

    Each case is solved from the accumulated pool of admissible states, and the
    sweep is repeated so that the case solved first is not penalised by having
    had no other solution to start from. The pool alone is not enough: it can
    only reach basins that some other cell already occupies. Every case is therefore also solved from ``random_starts``
    independent admissible draws, so that the reported estimate is the best of a
    multi-start rather than the state a sequential solve happened to select. The draws use a fixed seed, so the
    reported estimate is reproducible.
    """
    feedstocks = load_feedstocks()
    catalogue, relative_sd = load_gas_data()
    thermal_bases, runs_by_feedstock = {}, {}
    for name in FEEDSTOCKS:
        curve, runs = mean_thermogram(name)
        thermal_bases[name] = build_thermal_basis(feedstocks[name], curve, runs)
        runs_by_feedstock[name] = runs

    contexts, states, donors = {}, {}, {}
    for case in CASES:
        label = case_name(case)
        gas = catalogue[case]
        shared = dict(
            feedstock=feedstocks[case[0]],
            thermal=thermal_bases[case[0]],
            composition=gas.mean,
            composition_sd=gas.sd,
            alpha=ALPHA_TRANSFERRED,
        )
        donors[label] = solve(ModelContext(**shared, differential=False)).vector
        contexts[label] = ModelContext(**shared)

    for sweep in range(passes):
        for case in CASES:
            label = case_name(case)
            pool = [donors[label]] + [v.vector for v in states.values()]
            if label in states:
                pool.insert(0, states[label].vector)
            candidate = solve(contexts[label], start=pool)
            if label not in states or candidate.objective < states[label].objective:
                states[label] = candidate

    # Independent draws from the admissible domain, in the same way as the
    # multistart ensembles of ccpyro.identifiability. The generator lives with that ensemble and is
    # imported here rather than at module level, because that module imports
    # this one. The search is expensive, so its outcome is cached in
    # BEST_VECTORS and reloaded on later runs; passing random_starts=0 with a
    # populated cache reproduces the reported estimate without repeating it.
    for case in CASES:
        label = case_name(case)
        for vector in cached_starts(label):
            # The cached vector is considered both as it stands and as the start
            # of a further solve, because the solver first projects its start
            # onto the feasible set and can leave a converged state for a
            # slightly worse one.
            for candidate in (state_from_vector(vector, contexts[label]),
                              solve(contexts[label], start=vector, restarts=1)):
                if candidate is None:
                    continue
                if candidate.objective < states[label].objective:
                    states[label] = candidate

    if random_starts > 0:
        from .identifiability import random_admissible_start

        rng = np.random.default_rng(RANDOM_SEED)
        for case in CASES:
            label = case_name(case)
            context = contexts[label]
            for _ in range(random_starts):
                candidate = solve(context, start=random_admissible_start(context, rng),
                                  restarts=1)
                if candidate.feasibility < -1e-5:
                    continue
                if candidate.objective < states[label].objective:
                    states[label] = candidate
        store_best({case_name(case): states[case_name(case)] for case in CASES})

    return {
        "feedstocks": feedstocks,
        "thermal": thermal_bases,
        "runs": runs_by_feedstock,
        "gas": catalogue,
        "relative_sd": relative_sd,
        "contexts": contexts,
        "states": states,
        "donors": donors,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGDATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    built = build_cases()
    feedstocks = built["feedstocks"]
    gas_catalogue, relative_sd = built["gas"], built["relative_sd"]
    thermal_bases = built["thermal"]

    characterisation, calibration, yields, performance = [], [], [], []
    bounds, verifications, thermal_curves, interpretation = [], [], [], []
    references: dict[str, dict[str, np.ndarray]] = {}
    contexts = built["contexts"]
    states = built["states"]
    donors = built["donors"]

    for treatment in TREATMENTS:
        label = case_name(treatment)
        feedstock = feedstocks[treatment[0]]
        thermal = thermal_bases[treatment[0]]
        gas = gas_catalogue[treatment]
        context, state = contexts[label], states[label]
        integral = replace(context, differential=False)
        donor = unpack(donors[label], integral)
        donor.objective = objective(donors[label], integral)
        interpretation.append(
            {
                "treatment": label,
                "objective_integral": donor.objective,
                "objective_differential": state.objective,
                "x_c_800_integral": donor.x_c[-1],
                "x_c_800_differential": state.x_c[-1],
                "y_t_800_integral": donor.y_t[-1],
                "y_t_800_differential": state.y_t[-1],
            }
        )

        composition = feedstock.daf_composition
        characterisation.append(
            {
                "treatment": label,
                "feedstock": treatment[0],
                "densification": treatment[1],
                "moisture": feedstock.moisture,
                "ash": feedstock.ash,
                "volatile_matter": feedstock.volatile_matter,
                "fixed_carbon": feedstock.fixed_carbon,
                "C_daf": composition["C"],
                "H_daf": composition["H"],
                "O_daf": composition["O"],
                "N_daf": composition["N"],
                "x0": feedstock.x0,
                "y0": feedstock.y0,
                "z0": feedstock.z0,
                "MM_bio": feedstock.molar_mass,
                "HHV": feedstock.hhv,
                "closure_residual": feedstock.closure_residual,
                "FC_daf": feedstock.fixed_carbon_daf,
            }
        )

        shape = shape_from_parameters(context.alpha, context.beta, context.peak)
        calibration.append(
            {
                "treatment": label,
                "feedstock": treatment[0],
                "densification": treatment[1],
                "T_M": context.peak,
                "T_on": thermal.onset_temperature,
                "beta": context.beta,
                "alpha": context.alpha,
                "rho": shape["rho"],
                "T_L": shape["T_L"],
                "T_R": shape["T_R"],
                "A": state.parameters[1],
                "kappa": state.parameters[0],
                "a_H": state.parameters[2],
                "a_O": state.parameters[3],
                "floor_active_nodes": int(thermal.floor_active.sum()),
            }
        )

        verifications.append({"treatment": label, **verification(state, context)})
        frame = oxygen_bound(
            gas.mean,
            thermal.theta_nodes,
            feedstock.y0,
            feedstock.molar_mass,
            weighting=thermal.raw_nodes,
        )
        frame.insert(0, "treatment", label)
        frame["phi_tar_literature"] = tar_selectivity(
            T_NODES, AMPLITUDE_PRIOR, ALPHA_TRANSFERRED, 0.008, 338.16
        )
        frame["violates"] = frame["phi_tar_literature"] < frame["phi_tar_min"] - 1e-9
        bounds.append(frame)

        reference_record = collect(state, context)
        references[label] = reference_record
        draws = propagate(context, state, relative_sd, N_MONTE_CARLO, rng)

        summary = summarise(draws, reference_record)
        summary.insert(0, "treatment", label)
        dry = dry_basis(state, context)
        for key, values in dry.items():
            summary[f"{key}_dry"] = values
        summary["n_c"] = state.n_c
        summary["n_tar"] = state.n_t
        summary["n_CO"] = state.n_CO
        summary["n_CO2"] = state.n_CO2
        summary["n_CH4"] = state.n_CH4
        summary["n_H2"] = state.n_H2
        summary["n_H2O"] = state.n_H2O
        summary["phi_tar_model"] = state.phi_tar
        for index, species in enumerate(GAS_SPECIES):
            summary[f"phi_{species}_model"] = state.phi_model[:, index]
            summary[f"phi_{species}_exp"] = gas.mean[:, index]
            summary[f"phi_{species}_sd"] = gas.sd[:, index]
        yields.append(summary)

        indices = performance_indices(state, context)
        performance.append(
            pd.DataFrame(
                {
                    "treatment": label,
                    "T": T_NODES,
                    **{key: value for key, value in indices.items()},
                }
            )
        )

        thermal_curves.append(
            pd.DataFrame(
                {
                    "treatment": label,
                    "T": thermal.temperature_fine,
                    "theta": thermal.residue_fine,
                    "rate": thermal.rate_fine,
                }
            )
        )

    pd.DataFrame(characterisation).to_csv(RESULTS / "feedstock_characterisation.csv", index=False)
    pd.DataFrame(calibration).to_csv(RESULTS / "model_calibration.csv", index=False)
    pd.DataFrame(verifications).to_csv(RESULTS / "model_verification.csv", index=False)
    pd.DataFrame(interpretation).to_csv(RESULTS / "gc_interpretation.csv", index=False)
    pd.concat(yields, ignore_index=True).to_csv(RESULTS / "product_yields.csv", index=False)
    pd.concat(performance, ignore_index=True).to_csv(RESULTS / "performance_indices.csv", index=False)
    pd.concat(bounds, ignore_index=True).to_csv(RESULTS / "oxygen_bound.csv", index=False)
    pd.concat(thermal_curves, ignore_index=True).to_csv(FIGDATA / "thermal_curves.csv", index=False)


    reference_context = contexts["rice husk (pelletized)"]
    reference_state = states["rice husk (pelletized)"]
    sensitivity_to_alpha(reference_context, reference_state).to_csv(
        RESULTS / "sensitivity_alpha.csv", index=False
    )
    hessian_spectrum(reference_context, reference_state).to_csv(
        RESULTS / "identifiability.csv", index=False
    )
    structural_sensitivity(reference_context, reference_state).to_csv(
        RESULTS / "structural_sensitivity.csv", index=False
    )
    holdout = []
    for label, context in contexts.items():
        frame = holdout_validation(context, states[label])
        frame.insert(0, "treatment", label)
        holdout.append(frame)
    pd.concat(holdout).to_csv(RESULTS / "holdout_validation.csv", index=False)

    metadata = {
        "temperature_reference": T_REF,
        "temperature_nodes": T_NODES.tolist(),
        "monte_carlo_draws": N_MONTE_CARLO,
        "random_seed": RANDOM_SEED,
        "alpha": ALPHA_TRANSFERRED,
        "relative_sd_chromatography": dict(zip(GAS_SPECIES, relative_sd.round(5).tolist())),
        "objective_scale": OBJECTIVE_SCALE,
        "weights": OMEGA,
        "scales": SIGMA,
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    write_figure_data()


if __name__ == "__main__":
    main()
