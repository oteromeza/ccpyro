"""Energy balance of a small pyrolysis unit fed with the residues, from the estimated product distribution.

Per kilogram of residue as received, at the operating temperature T_op:
  heat demand  Q = [M * (h_fg + c_w * 75) + (1 - M) * c_p * (T_op - 25) + f_org * dH_r] / eta_th
  heat supply  tar first (eta_b), then char, then gas
  electricity  E_el = eta_e * (E_gas - gas burnt for heat) * (1 - aux)
The heating value of the tar and the char is obtained from their estimated CHxOy with the correlation of
Channiwala and Parikh (2002), less the latent heat of the water formed from their hydrogen. Uncertainty:
the product yields and compositions are sampled from the Monte Carlo summary of the framework (normal,
truncated at zero) and the technical parameters from the ranges below.

usage: python -m ccpyro.energy <product_yields.csv> <feedstock_characterisation.csv> <out.csv>
"""
import sys

import numpy as np
import pandas as pd

M = {"C": 12.011, "H": 1.008, "O": 15.999}
H_FG = 2.442            # MJ/kg, latent heat of water at 25 C
MOLAR_VOLUME = 0.022414   # Nm3/mol
LHV_MOLAR = {g: v * MOLAR_VOLUME for g, v in {"CO": 12.64, "H2": 10.80, "CH4": 35.80, "CO2": 0.0}.items()}  # MJ/mol
RANGES = {              # uniform ranges of the technical parameters
    "c_p": (1.3e-3, 1.7e-3),       # MJ/(kg K), dry solid and volatiles
    "dH_r": (0.0, 0.9),            # MJ/kg organic, endothermic enthalpy of pyrolysis
    "eta_th": (0.50, 0.75),        # thermal efficiency of the reactor, wall and flue-gas losses of the heating
                                   # jacket (Honus et al., 2014; Soponpongpipat et al., 2020; Fambri et al., 2024)
    "eta_b": (0.80, 0.90),         # efficiency of the burner, incomplete combustion and burner losses
                                   # (Punnarapong et al., 2017; Fambri et al., 2024)
    "eta_e": (0.20, 0.30),         # electrical efficiency of a small spark-ignition gas engine (Martinez et al., 2012)
    "aux": (0.08, 0.15),           # auxiliary consumption, fraction of the gross electricity (Patuzzi et al., 2021;
                                   # CERC, 2024)
}


def lhv_from_ratio(x, y):
    """LHV (MJ/kg) of a CHxOy lump from Channiwala and Parikh (2002), less the latent heat of its water."""
    mass = M["C"] + M["H"] * x + M["O"] * y
    c, h, o = 100 * M["C"] / mass, 100 * M["H"] * x / mass, 100 * M["O"] * y / mass
    hhv = 0.3491 * c + 1.1783 * h - 0.1034 * o
    return hhv - H_FG * (18.015 / 2.016) * h / 100


def unit_balance(row, feed, p):
    f_org = 1 - feed.moisture / 100 - feed.ash / 100
    mbio = feed.MM_bio
    gas_mol = {g: row[f"n_{g}"] for g in ("CO", "CO2", "CH4", "H2")}
    e_gas = f_org * 1000 / mbio * sum(gas_mol[g] * LHV_MOLAR[g] for g in gas_mol)   # MJ/kg a.r.
    e_tar = f_org * row.w_tar * lhv_from_ratio(row.x_t, row.y_t)
    e_char = f_org * row.w_char * lhv_from_ratio(row.x_c, row.y_c)
    q = (feed.moisture / 100 * (H_FG + 4.18e-3 * 75) + (1 - feed.moisture / 100) * p["c_p"] * (row["T"] - 25)
         + f_org * p["dH_r"]) / p["eta_th"]
    need = q / p["eta_b"]                     # fuel energy to be burnt
    from_tar = min(e_tar, need)
    from_char = min(e_char, need - from_tar)
    from_gas = min(e_gas, need - from_tar - from_char)
    e_el = p["eta_e"] * (e_gas - from_gas) * (1 - p["aux"])
    return {"heat_demand": q, "e_gas": e_gas, "e_tar": e_tar, "e_char": e_char,
            "tar_share_of_heat": from_tar / need, "char_used": from_char / e_char if e_char > 0 else 0.0,
            "electricity_kWh_t": e_el / 3.6 * 1000, "char_left": (e_char - from_char) / e_char if e_char else 0}


def electricity_draws(yields: pd.DataFrame, feeds: pd.DataFrame, treatment: str, T: float, draws: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo sample of the net electricity (kWh per tonne as received) of one feedstock at T."""
    row = yields[(yields["treatment"] == treatment) & (yields["T"] == T)].iloc[0]
    feed = feeds.loc[treatment]
    out = np.empty(draws)
    for i in range(draws):
        p = {k: rng.uniform(*v) for k, v in RANGES.items()}
        r = row.copy()
        for name in ("w_tar", "w_char", "x_t", "y_t", "x_c", "y_c"):
            r[name] = max(rng.normal(row[name], row.get(f"{name}_sd", 0.0)), 1e-6)
        scale = max(rng.normal(row.CGE, row.CGE_sd), 0.0) / row.CGE
        for gname in ("CO", "CO2", "CH4", "H2"):
            r[f"n_{gname}"] = row[f"n_{gname}"] * scale
        out[i] = unit_balance(r, feed, p)["electricity_kWh_t"]
    return out


if __name__ == "__main__":
    yields_path, feed_path, out_path = sys.argv[1:4]
    y = pd.read_csv(yields_path)
    feeds = pd.read_csv(feed_path).set_index("treatment")
    rng = np.random.default_rng(20261014)
    draws = 2000
    rows = []
    for (treatment, T), g in y.groupby(["treatment", "T"]):
        if T not in (500.0, 800.0):
            continue
        row = g.iloc[0]
        feed = feeds.loc[treatment]
        results = []
        for _ in range(draws):
            p = {k: rng.uniform(*v) for k, v in RANGES.items()}
            r = row.copy()
            for name in ("w_tar", "w_char", "x_t", "y_t", "x_c", "y_c"):
                sd = row.get(f"{name}_sd", 0.0)
                r[name] = max(rng.normal(row[name], sd), 1e-6)
            scale = max(rng.normal(row.CGE, row.CGE_sd), 0.0) / row.CGE      # gas energy through the CGE spread
            for gname in ("CO", "CO2", "CH4", "H2"):
                r[f"n_{gname}"] = row[f"n_{gname}"] * scale
            results.append(unit_balance(r, feed, p))
        frame = pd.DataFrame(results)
        summary = {"treatment": treatment, "T": T}
        for c in frame.columns:
            summary[c] = frame[c].median()
            summary[c + "_p05"] = frame[c].quantile(0.05)
            summary[c + "_p95"] = frame[c].quantile(0.95)
        summary["self_sufficient_with_tar"] = float(np.mean(frame["tar_share_of_heat"] >= 0.999))
        rows.append(summary)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    pd.set_option("display.width", 250)
    show = ["treatment", "T", "heat_demand", "e_gas", "e_tar", "e_char", "tar_share_of_heat",
            "self_sufficient_with_tar", "char_used", "electricity_kWh_t", "electricity_kWh_t_p05",
            "electricity_kWh_t_p95"]
    print(out[show].round(3).to_string(index=False))
