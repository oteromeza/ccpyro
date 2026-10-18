"""CRECK multistep mechanism of biomass pyrolysis (Debiagi et al., 2018, J. Anal. Appl. Pyrolysis 134,
326-335), read from the CHEMKIN file of its supplementary material, together with the characterisation of
the feedstock from its ultimate analysis through reference mixtures (Debiagi et al., 2015; Ranzi et al.,
2017).

The mechanism is integrated under a linear heating programme, and the products are mapped onto the lumps
of the conservation-constrained framework (char, tar, reaction water and CO, CO2, CH4, H2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

R_CAL = 1.987204  # cal/(mol K)
ATOMIC = {"C": 12.011, "H": 1.008, "O": 15.999}

# Elemental formulas (C, H, O). The metaplastic species G{X} are named XS in the CHEMKIN file;
# G{COH2}Stiff is COH2S and G{COH2}Loose is CH2OS, both with the formula CH2O.
FORMULA = {
    # reference species of the feedstock and solid intermediates
    "CELL": (6, 10, 5), "CELLA": (6, 10, 5),
    "GMSW": (5, 8, 4), "XYHW": (5, 8, 4), "XYGR": (5, 8, 4), "HCE1": (5, 8, 4), "HCE2": (5, 8, 4),
    "LIGC": (15, 14, 4), "LIGH": (22, 28, 9), "LIGO": (20, 22, 10),
    "LIGCC": (15, 14, 4), "LIGOH": (19, 22, 8), "LIG": (11, 12, 4),
    "TGL": (57, 100, 7), "TANN": (15, 12, 7), "ITANN": (8, 4, 4),
    "ACQUA": (0, 2, 1), "ASH": (0, 0, 0),
    "CHAR": (1, 0, 0),
    # metaplastic species
    "COH2S": (1, 2, 1), "CH2OS": (1, 2, 1), "CO2S": (1, 0, 2), "COS": (1, 0, 1), "CH3OHS": (1, 4, 1),
    "CH4S": (1, 4, 0), "C2H4S": (2, 4, 0), "C2H6S": (2, 6, 0), "C6H5OHS": (6, 6, 1), "H2S": (0, 2, 0),
    # volatile species
    "CO": (1, 0, 1), "CO2": (1, 0, 2), "CH4": (1, 4, 0), "H2": (0, 2, 0), "H2O": (0, 2, 1),
    "C2H4": (2, 4, 0), "C2H6": (2, 6, 0),
    "CH2O": (1, 2, 1), "CH3OH": (1, 4, 1), "HCOOH": (1, 2, 2), "CH3CHO": (2, 4, 1),
    "CH2OHCHO": (2, 4, 2), "CHOCHO": (2, 2, 2), "CH3CO2H": (2, 4, 2), "C2H5OH": (2, 6, 1),
    "C2H5CHO": (3, 6, 1), "C2H3CHO": (3, 4, 1), "CH2OHCH2CHO": (3, 6, 2),
    "C6H6O3": (6, 6, 3), "C6H10O5": (6, 10, 5), "C5H8O4": (5, 8, 4), "FURFURAL": (5, 4, 2),
    "C6H5OH": (6, 6, 1), "C6H5OCH3": (7, 8, 1), "CRESOL": (7, 8, 1), "VANILLIN": (8, 8, 3),
    "C24H28O4": (24, 28, 4), "MLINO": (19, 34, 2), "U2ME12": (13, 22, 2),
}

SOLID_REFERENCE = ("CELL", "GMSW", "XYHW", "XYGR", "LIGC", "LIGH", "LIGO", "TGL", "TANN", "ACQUA", "ASH")
SOLID_INTERMEDIATE = ("CELLA", "HCE1", "HCE2", "LIGCC", "LIGOH", "LIG", "ITANN", "CHAR")
METAPLASTIC = ("COH2S", "CH2OS", "CO2S", "COS", "CH3OHS", "CH4S", "C2H4S", "C2H6S", "C6H5OHS", "H2S")
SOLID = SOLID_REFERENCE + SOLID_INTERMEDIATE + METAPLASTIC
MEASURED_GAS = ("CO", "CO2", "CH4", "H2")
UNMEASURED_HYDROCARBONS = ("C2H4", "C2H6")


def molar_mass(name: str) -> float:
    c, h, o = FORMULA[name]
    return c * ATOMIC["C"] + h * ATOMIC["H"] + o * ATOMIC["O"]


@dataclass
class Reaction:
    reactant: str
    products: dict[str, float]
    A: float
    n: float
    E: float  # cal/mol
    text: str = ""

    def rate_constant(self, T: float) -> float:
        return self.A * T**self.n * np.exp(-self.E / (R_CAL * T))

    def imbalance(self) -> np.ndarray:
        """Atoms of C, H and O in the products minus those in the reactant."""
        lhs = np.array(FORMULA[self.reactant], float)
        rhs = sum(coef * np.array(FORMULA[s], float) for s, coef in self.products.items())
        return rhs - lhs


@dataclass
class Mechanism:
    reactions: list[Reaction]
    species: list[str] = field(default_factory=list)

    def __post_init__(self):
        names = set()
        for r in self.reactions:
            names.add(r.reactant)
            names.update(r.products)
        ordered = [s for s in SOLID if s in names] + sorted(names - set(SOLID))
        self.species = ordered
        self.index = {s: i for i, s in enumerate(self.species)}
        self.mw = np.array([molar_mass(s) for s in self.species])
        self.atoms = np.array([FORMULA[s] for s in self.species], float)  # (n_species, 3)
        n_s, n_r = len(self.species), len(self.reactions)
        self.nu = np.zeros((n_s, n_r))
        self.reactant_index = np.zeros(n_r, int)
        for j, r in enumerate(self.reactions):
            self.nu[self.index[r.reactant], j] -= 1.0
            for s, coef in r.products.items():
                self.nu[self.index[s], j] += coef
            self.reactant_index[j] = self.index[r.reactant]
        self.A = np.array([r.A for r in self.reactions])
        self.n = np.array([r.n for r in self.reactions])
        self.E = np.array([r.E for r in self.reactions])

    def k(self, T: float) -> np.ndarray:
        return self.A * T**self.n * np.exp(-self.E / (R_CAL * T))

    def jacobian(self, T: float) -> np.ndarray:
        """dn/dt = J n, with first-order reactions in their single reactant."""
        J = np.zeros((len(self.species), len(self.species)))
        k = self.k(T)
        for j, i in enumerate(self.reactant_index):
            J[:, i] += self.nu[:, j] * k[j]
        return J


def load_mechanism() -> Mechanism:
    """Mechanism of Debiagi et al. (2018) as shipped with the package (see creck_mechanism.py)."""
    from .creck_mechanism import REACTIONS
    return Mechanism([Reaction(reactant, dict(products), A, n, E, reactant)
                      for reactant, products, A, n, E in REACTIONS])


TERM = re.compile(r"^\s*([0-9]*\.?[0-9]*)\s*([A-Za-z][A-Za-z0-9_]*)\s*$")


def read_chemkin(path: str) -> Mechanism:
    text = open(path, encoding="latin-1").read()
    block = text[text.index("REACTIONS") + len("REACTIONS"): text.rindex("END")]
    reactions = []
    for raw in block.splitlines():
        line = raw.split("!")[0].strip()
        if not line or "=>" not in line:
            continue
        equation, *numbers = line.split()
        A, n, E = (float(v) for v in numbers[:3])
        lhs, rhs = equation.split("=>")
        products: dict[str, float] = {}
        for term in rhs.split("+"):
            match = TERM.match(term)
            if not match:
                raise ValueError(f"cannot parse term {term!r} in {equation}")
            coef = float(match.group(1)) if match.group(1) not in ("", ".") else 1.0
            products[match.group(2)] = products.get(match.group(2), 0.0) + coef
        reactions.append(Reaction(lhs.strip(), products, A, n, E, equation))
    return Mechanism(reactions)


DEFAULT_SPLITTING = {"alpha": 0.6, "beta": 0.8, "gamma": 0.8, "delta": 1.0, "epsilon": 1.0}
CHAR_SPECIES = ("CELL", "HCE", "LIGC", "LIGH", "LIGO", "TANN", "TGL")


def reference_mixtures(alpha, beta, gamma, delta, epsilon):
    """Molar composition of RM-1, RM-2 and RM-3 over CELL, HCE, LIGC, LIGH, LIGO, TANN, TGL."""
    rm1 = np.array([alpha, 1 - alpha, 0, 0, 0, 0, 0])
    rm2 = np.array([0, 0, (1 - beta) * delta, beta * delta, 0, 0, 1 - delta])
    rm3 = np.array([0, 0, (1 - gamma) * epsilon, 0, gamma * epsilon, 1 - epsilon, 0])
    return np.vstack([rm1, rm2, rm3])


def _species_formula():
    f = dict(FORMULA)
    f["HCE"] = (5, 8, 4)
    atoms = np.array([f[s] for s in CHAR_SPECIES], float)
    mw = atoms @ np.array([ATOMIC["C"], ATOMIC["H"], ATOMIC["O"]])
    return atoms, mw


def characterise(w_C: float, w_H: float, w_O: float | None = None, **splitting) -> dict:
    """Mass fractions of the seven reference species reproducing the C/H/O mass fractions exactly.

    Returns the reference-mixture and species mass fractions (daf). Negative values mean that the feedstock
    lies outside the triangle spanned by the reference mixtures with these splitting parameters.
    """
    p = {**DEFAULT_SPLITTING, **splitting}
    if w_O is None:
        w_O = 1.0 - w_C - w_H
    total = w_C + w_H + w_O
    target = np.array([w_C, w_H, w_O]) / total
    atoms, mw = _species_formula()
    rm_molar = reference_mixtures(**p)                         # (3, 7) mole fractions
    rm_mass_species = rm_molar * mw                            # grams of each species per mol of RM
    rm_mw = rm_mass_species.sum(axis=1)
    rm_element_mass = (rm_molar @ atoms) * np.array([ATOMIC["C"], ATOMIC["H"], ATOMIC["O"]])
    rm_element_fraction = rm_element_mass / rm_mw[:, None]     # (3, 3): rows RM, columns C/H/O
    y_rm = np.linalg.solve(rm_element_fraction.T, target)      # mass fractions of RM-1..3
    species_mass = (y_rm / rm_mw) @ rm_mass_species             # mass of each species per unit mass
    return {"rm": y_rm, "species": dict(zip(CHAR_SPECIES, species_mass)), "splitting": p,
            "rm_elements": rm_element_fraction}


def characterise_admissible(w_C: float, w_H: float, w_O: float | None = None) -> dict:
    """Reference-mixture characterisation with non-negative mixtures, following Ranzi et al. (2017).

    1. Default splitting parameters.
    2. Otherwise, the extractive contents are adjusted (delta for TGL, epsilon for TANN), keeping the
       other parameters at their defaults, with the smallest change from the defaults.
    3. Otherwise, all five parameters are adjusted within their ranges, with the smallest change.
    4. If no admissible set exists, the feedstock lies outside the domain of the reference species, and the
       nearest representable composition is returned (``exact`` = False).
    """
    if w_O is None:
        w_O = 1.0 - w_C - w_H
    names = list(DEFAULT_SPLITTING)
    default = np.array([DEFAULT_SPLITTING[k] for k in names])

    def rm_of(v):
        try:
            return characterise(w_C, w_H, w_O, **dict(zip(names, v)))["rm"]
        except np.linalg.LinAlgError:
            return -np.ones(3)

    base = characterise(w_C, w_H, w_O)
    if np.all(base["rm"] >= -1e-12):
        base["exact"], base["step"] = True, 1
        return base

    stages = (
        (2, {"alpha": (0.6, 0.6), "beta": (0.8, 0.8), "gamma": (0.8, 0.8), "delta": (0.0, 1.0),
             "epsilon": (0.0, 1.0)}),
        (3, {"alpha": (0.4, 0.8), "beta": (0.0, 1.0), "gamma": (0.0, 1.0), "delta": (0.0, 1.0),
             "epsilon": (0.0, 1.0)}),
    )
    for step, ranges in stages:
        box = [ranges[k] for k in names]
        rng = np.random.default_rng(step)
        starts = [np.clip(default, [b[0] for b in box], [b[1] for b in box])]
        starts += [np.array([rng.uniform(a, b) for a, b in box]) for _ in range(60)]
        best = None
        for start in starts:
            res = minimize(lambda v: np.sum((v - default) ** 2), start, method="SLSQP", bounds=box,
                           constraints=[{"type": "ineq", "fun": rm_of}],
                           options={"ftol": 1e-12, "maxiter": 500})
            if np.all(rm_of(res.x) >= -1e-9) and (best is None or res.fun < best.fun):
                best = res
        if best is not None:
            out = characterise(w_C, w_H, w_O, **dict(zip(names, best.x)))
            out["exact"], out["step"] = True, step
            return out

    # Step 4: nearest representable feedstock within the reference-mixture framework. The mixture fractions
    # are free (non-negative, summing to one), the C/H/O mismatch is minimised, and a small penalty keeps
    # the splitting parameters near their defaults, so that the composition remains a realistic mixture of
    # holocellulose, lignins and extractives.
    target = np.array([w_C, w_H, w_O]) / (w_C + w_H + w_O)
    scale = np.array([0.5, 0.06, 0.44])
    ranges = {"alpha": (0.4, 0.8), "beta": (0.0, 1.0), "gamma": (0.0, 1.0), "delta": (0.0, 1.0),
              "epsilon": (0.0, 1.0)}
    box = [ranges[k] for k in names] + [(0.0, 1.0)] * 3

    def elements_of(v):
        p = dict(zip(names, v[:5]))
        atoms, mw = _species_formula()
        rm_molar = reference_mixtures(**p)
        rm_mw = (rm_molar * mw).sum(axis=1)
        rm_el = (rm_molar @ atoms) * np.array([ATOMIC["C"], ATOMIC["H"], ATOMIC["O"]]) / rm_mw[:, None]
        return v[5:] @ rm_el

    def merit(v):
        return np.sum(((elements_of(v) - target) / scale) ** 2) + 1e-4 * np.sum((v[:5] - default) ** 2)

    rng = np.random.default_rng(4)
    best = None
    for _ in range(80):
        start = np.concatenate([[rng.uniform(a, b) for a, b in box[:5]], rng.dirichlet(np.ones(3))])
        res = minimize(merit, start, method="SLSQP", bounds=box,
                       constraints=[{"type": "eq", "fun": lambda v: v[5:].sum() - 1.0}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res
    v = best.x
    p = dict(zip(names, v[:5]))
    atoms, mw = _species_formula()
    rm_molar = reference_mixtures(**p)
    rm_mass_species = rm_molar * mw
    rm_mw = rm_mass_species.sum(axis=1)
    species_mass = (v[5:] / rm_mw) @ rm_mass_species
    return {"rm": v[5:], "species": dict(zip(CHAR_SPECIES, species_mass)), "splitting": p, "exact": False,
            "step": 4, "represented": elements_of(v), "target": target}


def characterise_biochemical(w_C: float, w_H: float, w_O: float, cellulose: float, hemicellulose: float,
                             lignin: float, extractives: float) -> dict:
    """Reference species from a measured biochemical analysis (for example, van Soest).

    The four fractions are normalised to the dry, ash-free basis. Cellulose and hemicellulose are assigned directly
    to CELL and HCE; the lignin is split into LIGC, LIGH and LIGO and the extractives into TGL and TANN so that the
    elemental composition is as close as possible to the ultimate analysis, with a small penalty that keeps the
    splits near equal shares where the analysis does not determine them. The remaining difference with the
    ultimate analysis is returned in ``represented`` and measures the consistency of the two analyses.
    """
    total = cellulose + hemicellulose + lignin + extractives
    cel, hem, lig, ext = (v / total for v in (cellulose, hemicellulose, lignin, extractives))
    target = np.array([w_C, w_H, w_O]) / (w_C + w_H + w_O)
    scale = np.array([0.5, 0.06, 0.44])
    mass = np.array([ATOMIC["C"], ATOMIC["H"], ATOMIC["O"]])

    def fractions(name):
        f = np.array(FORMULA[name], float) * mass
        return f / f.sum()

    E = {name: fractions(name) for name in ("CELL", "XYGR", "LIGC", "LIGH", "LIGO", "TGL", "TANN")}

    def species(u):
        return {"CELL": cel, "HCE": hem, "LIGC": lig * u[0], "LIGH": lig * u[1], "LIGO": lig * u[2],
                "TGL": ext * u[3], "TANN": ext * (1.0 - u[3])}

    def elements(u):
        sp = species(u)
        return sum(sp[k] * E["XYGR" if k == "HCE" else k] for k in sp)

    prior = np.array([1 / 3, 1 / 3, 1 / 3, 0.5])

    def merit(u):
        return np.sum(((elements(u) - target) / scale) ** 2) + 1e-4 * np.sum((u - prior) ** 2)

    best = None
    rng = np.random.default_rng(5)
    for start in [prior] + [np.concatenate([rng.dirichlet(np.ones(3)), [rng.uniform()]]) for _ in range(30)]:
        res = minimize(merit, start, method="SLSQP", bounds=[(0.0, 1.0)] * 4,
                       constraints=[{"type": "eq", "fun": lambda u: u[:3].sum() - 1.0}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        if best is None or res.fun < best.fun:
            best = res
    u = best.x
    return {"species": species(u), "represented": elements(u), "target": target, "exact": False,
            "step": "biochemical", "splitting": {"LIGC": float(u[0]), "LIGH": float(u[1]), "LIGO": float(u[2]),
                                                  "TGL_share_of_extractives": float(u[3])},
            "biochemical_daf": {"cellulose": cel, "hemicellulose": hem, "lignin": lig, "extractives": ext}}


@dataclass
class Programme:
    T0: float = 303.15        # K
    rate: float = 0.5         # K/s (30 K/min)
    T_end: float = 1113.15    # K (840 C)

    def T(self, t):
        return np.minimum(self.T0 + self.rate * t, self.T_end)

    def time_at(self, T):
        return (T - self.T0) / self.rate


def initial_moles(mech: Mechanism, species_mass: dict, hemicellulose: str = "XYGR") -> np.ndarray:
    """Moles of each reference species per gram of dry, ash-free feedstock."""
    n0 = np.zeros(len(mech.species))
    for name, mass in species_mass.items():
        target = hemicellulose if name == "HCE" else name
        if mass > 0:
            n0[mech.index[target]] += mass / molar_mass(target)
    return n0


def integrate(mech: Mechanism, n0: np.ndarray, programme: Programme, T_eval_C, rtol=1e-9, atol=1e-14):
    t_eval = np.array([programme.time_at(T + 273.15) for T in T_eval_C])
    t_end = t_eval.max()

    def rhs(t, n):
        return mech.jacobian(programme.T(t)) @ n

    def jac(t, n):
        return mech.jacobian(programme.T(t))

    sol = solve_ivp(rhs, (0.0, t_end), n0, method="Radau", jac=jac, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T  # (n_eval, n_species)


def lumps(mech: Mechanism, n: np.ndarray) -> dict:
    """Map one state (moles per gram of daf feedstock) onto the lumps of the framework.

    Amounts are referred to one mole of feedstock carbon, as in the framework. The tar includes every
    volatile species other than water and the four measured gases; the C2 hydrocarbons, which the
    chromatograph does not quantify, are also reported separately.
    """
    idx = mech.index
    atoms = mech.atoms
    solid = [s for s in mech.species if s in SOLID and s not in ("ACQUA", "ASH")]
    volatile = [s for s in mech.species if s not in SOLID]
    measured = list(MEASURED_GAS)
    tar = [s for s in volatile if s not in measured and s != "H2O"]
    tar_condensable = [s for s in tar if s not in UNMEASURED_HYDROCARBONS]

    def elements(names):
        return sum(n[idx[s]] * atoms[idx[s]] for s in names) if names else np.zeros(3)

    def mass(names):
        return sum(n[idx[s]] * mech.mw[idx[s]] for s in names) if names else 0.0

    out = {}
    out["solid_elements"] = elements(solid)
    out["tar_elements"] = elements(tar)
    out["tar_condensable_elements"] = elements(tar_condensable)
    out["gas"] = {s: n[idx[s]] for s in measured}
    out["C2"] = {s: n[idx[s]] for s in UNMEASURED_HYDROCARBONS}
    out["water"] = n[idx["H2O"]]
    out["mass"] = {"solid": mass(solid), "tar": mass(tar), "tar_condensable": mass(tar_condensable),
                   "water": n[idx["H2O"]] * mech.mw[idx["H2O"]], "gas": mass(measured)}
    return out


NODES_C = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0])
T_REF_C = 200.0


def run_benchmark(feedstocks: dict | None = None, programme: Programme | None = None, grid_step: float = 5.0,
                  basis: str = "ultimate"):
    """Characterise each feedstock, integrate the mechanism and map the products onto the lumps.

    ``feedstocks`` maps a name to the carbon, hydrogen and oxygen mass fractions on the dry, ash-free basis
    (nitrogen omitted, as in the framework); by default they are read from the data of the package. With
    ``basis="ultimate"`` the reference species follow from the ultimate analysis through reference
    mixtures; with ``"biochemical"`` they follow from the measured cellulose, hemicellulose, lignin and extractives
    of the package data. Returns the table at the temperature nodes, the residual-mass curve and the
    characterisation of each feedstock.
    """
    import pandas as pd

    from .model import GAS_LABEL, load_biochemical, load_feedstocks

    if feedstocks is None:
        feedstocks = {}
        for name, feed in load_feedstocks().items():
            c = feed.daf_composition
            feedstocks[name] = (c["C"], c["H"], c["O"])
    if basis not in ("ultimate", "biochemical"):
        raise ValueError(f"unknown basis of the characterisation: {basis}")
    biochemical = load_biochemical() if basis == "biochemical" else None
    mech = load_mechanism()
    programme = Programme() if programme is None else programme
    grid = np.arange(35.0, 840.0 + 1e-9, grid_step)
    gases = list(MEASURED_GAS)
    rows, curves, characterisation = [], [], {}
    for name, cho in feedstocks.items():
        w = np.asarray(cho, float) / float(np.sum(cho))
        if biochemical is None:
            ch = characterise_admissible(*w)
        else:
            b = biochemical.loc[GAS_LABEL[name]]
            ch = characterise_biochemical(*w, b["Cellulose"], b["Hemicellulose"], b["Lignin"], b["Extractives"])
        characterisation[name] = {
            "basis": basis, "step": ch["step"], "exact": bool(ch["exact"]),
            "splitting": {k: float(v) for k, v in ch["splitting"].items()},
            "species": {k: float(max(v, 0.0)) for k, v in ch["species"].items()},
            "target_CHO": [float(v) for v in w],
            "represented_CHO": [float(v) for v in ch.get("represented", w)],
        }
        n0 = initial_moles(mech, {k: max(v, 0.0) for k, v in ch["species"].items()}, hemicellulose="XYGR")
        carbon0 = float(n0 @ mech.atoms[:, 0])
        mass0 = float(n0 @ mech.mw)
        temps = np.unique(np.concatenate([grid, NODES_C, [T_REF_C]]))
        states = integrate(mech, n0, programme, temps)
        lumped = {float(T): lumps(mech, states[k]) for k, T in enumerate(temps)}
        solid_ref = lumped[T_REF_C]["mass"]["solid"]
        for T, L in lumped.items():
            curves.append({"feedstock": name, "T": T, "solid_daf": L["mass"]["solid"] / mass0,
                           "theta": L["mass"]["solid"] / solid_ref})
        previous = {g: 0.0 for g in gases}
        for T in NODES_C:
            L = lumped[float(T)]
            se, te, tc = L["solid_elements"], L["tar_elements"], L["tar_condensable_elements"]
            gas = {g: L["gas"][g] / carbon0 for g in gases}
            released = {g: gas[g] - previous[g] for g in gases}
            total = sum(released.values())
            previous = gas
            rows.append({
                "feedstock": name, "T": T,
                "theta": L["mass"]["solid"] / solid_ref,
                "w_char": L["mass"]["solid"] / mass0, "w_tar": L["mass"]["tar"] / mass0,
                "w_tar_condensable": L["mass"]["tar_condensable"] / mass0,
                "w_water": L["mass"]["water"] / mass0, "w_gas": L["mass"]["gas"] / mass0,
                "n_c": se[0] / carbon0, "x_c": se[1] / se[0], "y_c": se[2] / se[0],
                "n_t": te[0] / carbon0, "x_t": te[1] / te[0], "y_t": te[2] / te[0],
                "x_t_condensable": tc[1] / tc[0], "y_t_condensable": tc[2] / tc[0],
                "n_C2": (2 * L["C2"]["C2H4"] + 2 * L["C2"]["C2H6"]) / carbon0,
                "n_W": L["water"] / carbon0,
                "closure": float(np.abs(states[list(temps).index(T)] @ mech.atoms - n0 @ mech.atoms).max()),
                **{f"n_{g}": gas[g] for g in gases},
                **{f"phi_{g}": released[g] / total if total > 0 else np.nan for g in gases},
            })
    return pd.DataFrame(rows), pd.DataFrame(curves), characterisation


def compare_with_framework(nodes, yields):
    """Join the CRECK nodes with the estimates of the framework (``product_yields.csv``)."""
    est = yields.copy()
    est["feedstock"] = est["treatment"].str.replace(" (pelletized)", "", regex=False)
    cols = ["w_char", "w_tar", "w_water", "w_gas", "x_c", "y_c", "x_t", "y_t"]
    return nodes.merge(est[["feedstock", "T"] + cols + [f"phi_{g}_exp" for g in MEASURED_GAS] + ["n_tar"]],
                       on=["feedstock", "T"], suffixes=("_creck", "_est"))


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out.mkdir(parents=True, exist_ok=True)
    nodes, curves, characterisation = run_benchmark()
    nodes.to_csv(out / "creck_nodes.csv", index=False)
    curves.to_csv(out / "creck_thermogram.csv", index=False)
    (out / "creck_characterisation.json").write_text(json.dumps(characterisation, indent=2))
    if len(sys.argv) > 2:
        compare_with_framework(nodes, pd.read_csv(sys.argv[2])).to_csv(out / "creck_vs_framework.csv", index=False)
    print(nodes.round(4).to_string(index=False))
