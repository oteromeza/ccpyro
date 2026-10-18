"""Balance-and-correlation benchmark: constant tar formula with penalised balances.

The tar is represented by the constant formula C6H6.2O0.2 (Thunman et al., 2001), and the carbon, hydrogen and
oxygen balances enter the least-squares objective as weighted residuals, as in the classical formulation of this
group of models. Two variants are provided.

A  The char composition follows the correlations of Neves et al. (2011), as reported by Debiagi et al. (2018),
   Eqs. (2)-(4), at the temperature of the node.
B  The char composition follows the carbonisation closure of the framework, with fitted exponents.

In both variants, the solid, gas, water and selectivity blocks, their scales and weights and the parameter priors
are those of the framework, and only bounds are imposed (no admissibility constraints). A third check keeps the
exact elimination of the balances with the constant tar formula and determines whether an admissible state exists.

usage: python -m ccpyro.correlations [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from . import model as mp

X_T, Y_T = 6.2 / 6.0, 0.2 / 6.0           # Thunman et al. (2001)
T = mp.T_NODES
N = mp.N_NODES
MC, MH, MO = mp.M["C"], mp.M["H"], mp.M["O"]
VARIABLES = ("n_c", "n_t", "n_W", "n_CO", "n_CO2", "n_CH4", "n_H2")
NV = len(VARIABLES)
SIGMA_BALANCE = (0.001, 0.01, 0.05)       # tolerances of the penalised balances, mol per mol of carbon
SEED = 20261014


def neves_char(temperature_c):
    """Atomic H/C and O/C of the char from the mass-fraction correlations of Neves et al. (2011)."""
    y_c = 0.93 - 0.92 * np.exp(-0.0042 * temperature_c)
    y_o = 0.07 + 0.85 * np.exp(-0.0048 * temperature_c)
    y_h = -0.0041 + 0.1 * np.exp(-0.0024 * temperature_c)
    return (y_h / MH) / (y_c / MC), (y_o / MO) / (y_c / MC)


def unpack(v, ctx, variant):
    """Streams, compositions and balance residuals of a decision vector of the benchmark."""
    block = v[: N * NV].reshape(N, NV)
    g = v[N * NV:]
    kappa, amplitude = g[0], g[1]
    if variant == "A":
        x_c, y_c = neves_char(T)
        a_H = a_O = np.nan
    else:
        a_H, a_O = g[2], g[3]
        x_c, y_c = ctx.x0 * ctx.theta**a_H, ctx.y0 * ctx.theta**a_O
    n_c, n_t, n_W, n_CO, n_CO2, n_CH4, n_H2 = block.T
    gas = np.column_stack([n_CO, n_CO2, n_CH4, n_H2])
    released = np.vstack([gas[0], np.diff(gas, axis=0)])
    positive = np.maximum(released, 0.0)
    phi = positive / np.maximum(positive.sum(axis=1, keepdims=True), 1e-9)
    m_char = MC + MH * x_c + MO * y_c
    m_tar = MC + MH * X_T + MO * Y_T
    w_char = n_c * m_char / ctx.molar_mass
    w_tar = n_t * m_tar / ctx.molar_mass
    w_water = n_W * mp.MM["H2O"] / ctx.molar_mass
    w_gas = (gas @ mp.MM_GAS) / ctx.molar_mass
    bal_C = 1.0 - (n_c + n_t + n_CO + n_CO2 + n_CH4)
    bal_H = ctx.x0 - (x_c * n_c + X_T * n_t + 2 * n_W + 4 * n_CH4 + 2 * n_H2)
    bal_O = ctx.y0 - (y_c * n_c + Y_T * n_t + n_W + n_CO + 2 * n_CO2)
    return dict(n_c=n_c, n_t=n_t, n_W=n_W, n_CO=n_CO, n_CO2=n_CO2, n_CH4=n_CH4, n_H2=n_H2, x_c=x_c, y_c=y_c,
                phi=phi, w_char=w_char, w_tar=w_tar, w_water=w_water, w_gas=w_gas,
                bal=np.column_stack([bal_C, bal_H, bal_O]), kappa=kappa, A=amplitude, a_H=a_H, a_O=a_O)


def _blocks(v, ctx, variant, sigma_bal):
    s = unpack(v, ctx, variant)
    ratio = s["kappa"] * mp.MM["CO2"] / mp.MM["H2O"]
    water_target = ratio / (1.0 + ratio)
    water_model = s["n_W"] / np.maximum(s["n_W"] + s["n_CO2"], 1e-12)
    tar_target = mp.tar_selectivity(T, s["A"], ctx.alpha, ctx.beta, ctx.peak)
    phi_tar = s["w_tar"] / np.maximum(s["w_tar"] + s["w_gas"], 1e-12)
    prior_mean = mp.PRIOR_MEAN[:2] if variant == "A" else mp.PRIOR_MEAN
    prior_sd = mp.PRIOR_SD[:2] if variant == "A" else mp.PRIOR_SD
    params = np.array([s["kappa"], s["A"]] + ([] if variant == "A" else [s["a_H"], s["a_O"]]))
    return {
        "solid": np.sqrt(mp.OMEGA["tga"]) * (s["w_char"] - ctx.theta) / ctx.theta_scale,
        "gas": np.sqrt(mp.OMEGA["gc"]) * ((s["phi"] - ctx.composition) / ctx.composition_sd).ravel(),
        "water_closure": np.sqrt(mp.OMEGA["water"]) * (water_model - water_target) / mp.SIGMA["water"],
        "tar_closure": np.sqrt(mp.OMEGA["tar"]) * (phi_tar - tar_target) / mp.SIGMA["tar"],
        "balances": (s["bal"] / sigma_bal).ravel(),
        "prior": np.sqrt(mp.OMEGA["prior"]) * (params - prior_mean) / prior_sd,
    }


def terms(v, ctx, variant, sigma_bal):
    """Contribution of each block to the objective of the benchmark."""
    return {key: float(np.sum(r**2)) for key, r in _blocks(v, ctx, variant, sigma_bal).items()}


def residuals(v, ctx, variant, sigma_bal):
    """Standardised residual vector whose sum of squares equals the bracket of the objective."""
    return np.concatenate(list(_blocks(v, ctx, variant, sigma_bal).values()))


def solve(ctx, variant, sigma_bal, starts=8, rng=None):
    """Least-squares fit (trust-region reflective) from the point estimate of the framework and random starts."""
    rng = np.random.default_rng(SEED) if rng is None else rng
    n_glob = 2 if variant == "A" else 4
    lower = np.concatenate([np.zeros(N * NV), mp.PARAM_LOWER[:n_glob]])
    upper = np.concatenate([np.tile([1.0, 1.0, 2.0, 0.9, 0.9, 0.9, 2.0], N), mp.PARAM_UPPER[:n_glob]])
    best = None
    for i in range(starts):
        v0 = rng.uniform(lower, upper)
        v0[: N * NV] *= 0.3
        if i == 0:   # start built from the point estimate of the framework
            ref = ctx.reference
            v0[: N * NV] = np.column_stack([ref.n_c, ref.n_t, ref.n_H2O, ref.n_CO, ref.n_CO2, ref.n_CH4,
                                            ref.n_H2]).reshape(-1)
            v0[N * NV:] = ref.parameters[:n_glob]
        v0 = np.clip(v0, lower + 1e-9, upper - 1e-9)
        res = least_squares(residuals, v0, bounds=(lower, upper), args=(ctx, variant, sigma_bal),
                            method="trf", x_scale="jac", max_nfev=4000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        value = 2.0 * res.cost / mp.OBJECTIVE_SCALE   # least_squares reports half the sum of squares
        if best is None or value < best[1]:
            best = (res.x, value)
    return best


def exact_with_constant_formula(ctx, rng=None):
    """Framework with the tar fixed to C6H6.2O0.2: does an admissible state compatible with the data exist?"""
    rng = np.random.default_rng(SEED) if rng is None else rng
    lower, upper = ctx.lower.copy(), ctx.upper.copy()
    for k in range(N):
        for name, value in (("x_t", X_T), ("y_t", Y_T)):
            p = k * mp.N_FREE + mp.IDX[name]
            lower[p] = upper[p] = value

    def g(v):   # admissibility without the ordering and monotonicity of the tar, which no longer apply
        s = mp.unpack(v, ctx)
        incr = [np.concatenate([[q[0]], np.diff(q)]) for q in (s.n_CO, s.n_CO2, s.n_CH4, s.n_H2, s.n_H2O)]
        return np.concatenate([s.n_t, s.n_H2O, s.n_H2, -np.diff(np.concatenate([[1.0], s.n_c])), *incr])

    def violation(v):
        return float(np.sum(np.minimum(0.0, g(v)) ** 2))

    best = None
    for i in range(8):
        v0 = ctx.reference.vector.copy() if i == 0 else mp.initial_point(ctx) * rng.uniform(0.7, 1.3, ctx.lower.size)
        v0 = np.clip(v0, lower + 1e-12, upper - 1e-12)
        res = minimize(violation, v0, method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    worst = float(np.min(g(best.x)))
    fit = None
    if worst > -1e-6:
        res = minimize(mp.objective, best.x, args=(ctx,), method="SLSQP", bounds=list(zip(lower, upper)),
                       constraints=[{"type": "ineq", "fun": g}], options={"maxiter": 700, "ftol": 1e-12})
        fit = (res.fun, float(np.min(g(res.x))), mp.unpack(res.x, ctx))
    return worst, mp.unpack(best.x, ctx), fit


def context_with_reference(case, best_vectors=None):
    """Context of one cell with the point estimate of the framework attached as ``reference``."""
    feedstocks = mp.load_feedstocks()
    catalogue, _ = mp.load_gas_data()
    curve, runs = mp.mean_thermogram(case[0])
    thermal = mp.build_thermal_basis(feedstocks[case[0]], curve, runs)
    gas = catalogue[case]
    ctx = mp.ModelContext(feedstock=feedstocks[case[0]], thermal=thermal, composition=gas.mean,
                          composition_sd=gas.sd, alpha=mp.ALPHA_TRANSFERRED)
    vectors = np.load(mp.BEST_VECTORS) if best_vectors is None else best_vectors
    label = mp.case_name(case)
    ref = mp.unpack(vectors[label], ctx)
    ref.vector = vectors[label]
    ctx.reference = ref
    return ctx


def run(out_dir, cases=None, starts=8):
    """Fit both variants at the three tolerances for each cell and write the comparison tables."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, checks = [], []
    for case in (mp.CASES if cases is None else cases):
        ctx = context_with_reference(case)
        ref = ctx.reference
        for k in range(N):
            rows.append({"feedstock": case[0], "model": "framework", "sigma_bal": 0.0, "T": T[k],
                         "w_char": ref.w_char[k], "w_tar": ref.w_tar[k], "w_water": ref.w_water[k],
                         "w_gas": ref.w_gas[k], "bal_C": 0.0, "bal_H": 0.0, "bal_O": 0.0,
                         "gas_rmse": float(np.sqrt(np.mean((ref.phi_model[k] - ctx.composition[k]) ** 2))),
                         "solid_res": ref.w_char[k] - ctx.theta[k], "n_H2": ref.n_H2[k], "n_W": ref.n_H2O[k],
                         "x_c": ref.x_c[k], "y_c": ref.y_c[k], "x_t": ref.x_t[k], "y_t": ref.y_t[k]})
        for variant in ("A", "B"):
            for sigma_bal in SIGMA_BALANCE:
                v, _ = solve(ctx, variant, sigma_bal, starts=starts, rng=rng)
                s = unpack(v, ctx, variant)
                parts = terms(v, ctx, variant, sigma_bal)
                for k in range(N):
                    rows.append({"feedstock": case[0], "model": f"constant formula {variant}",
                                 "sigma_bal": sigma_bal, "T": T[k], "w_char": s["w_char"][k],
                                 "w_tar": s["w_tar"][k], "w_water": s["w_water"][k], "w_gas": s["w_gas"][k],
                                 "bal_C": s["bal"][k, 0], "bal_H": s["bal"][k, 1], "bal_O": s["bal"][k, 2],
                                 "gas_rmse": float(np.sqrt(np.mean((s["phi"][k] - ctx.composition[k]) ** 2))),
                                 "solid_res": s["w_char"][k] - ctx.theta[k], "n_H2": s["n_H2"][k],
                                 "n_W": s["n_W"][k], "x_c": float(np.atleast_1d(s["x_c"])[k]),
                                 "y_c": float(np.atleast_1d(s["y_c"])[k]), "x_t": X_T, "y_t": Y_T,
                                 **{f"term_{t}": val for t, val in parts.items()}})
        worst, _, fit = exact_with_constant_formula(ctx, rng=rng)
        checks.append({"feedstock": case[0], "max_violation": worst, "feasible": worst > -1e-6,
                       "fit_objective": None if fit is None else fit[0]})
    table, check = pd.DataFrame(rows), pd.DataFrame(checks)
    table.to_csv(out / "fourth_group.csv", index=False)
    check.to_csv(out / "fourth_group_exact_check.csv", index=False)
    return table, check


if __name__ == "__main__":
    table, check = run(sys.argv[1] if len(sys.argv) > 1 else mp.RESULTS)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table[["feedstock", "model", "sigma_bal", "T", "w_char", "w_tar", "w_water", "w_gas", "bal_C", "bal_H",
                 "bal_O", "gas_rmse", "solid_res"]].round(4).to_string(index=False))
    print(check.to_string(index=False))
