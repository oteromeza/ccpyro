"""Electricity that rice husk and corncob could supply to rural populations without access, by country.

Residue supply = production (paddy rice, maize; mean 2022-2024) x residue-to-product ratio x available fraction.
Electricity  = supply x net electricity per tonne from the energy balance of the unit (module ``energy``).
Demand       = rural population without access x basic household consumption (Multi-Tier Framework) / household size.
Every uncertain input is sampled in a Monte Carlo of N draws; results are medians with 5-95 % intervals.

The public data ship with the package in ``data/global`` and are refreshed with ``tools/fetch_global_data.py``.

usage: python -m ccpyro.global_potential <product_yields.csv> <feedstock_characterisation.csv> <T_op> <out_prefix>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .energy import electricity_draws

GLOBAL_DATA = Path(__file__).resolve().parent / "data" / "global"
PRODUCTION = GLOBAL_DATA / "faostat_production_2022_2024.csv"
WORLD_BANK = GLOBAL_DATA / "worldbank_rural_access.csv"

N_DRAWS = 4000
SEED = 20261014
RPR = {"rice": (0.20, 0.30), "maize": (0.20, 0.30)}      # kg residue per kg product (Koopmans and Koppejan, 1997)
AVAILABLE = (0.3, 0.7)                                    # fraction not claimed by other uses (assumption)
HOUSEHOLD = (4.0, 6.0)                                    # persons per rural household (assumption)
TIERS = {"Tier 2": 73.0, "Tier 3": 365.0}                 # kWh per household and year (Bhatia and Angelou, 2015)
DIESEL = (0.30, 0.45)                                     # L/kWh of small gensets at partial load
CO2_PER_LITRE = 2.68                                      # kg CO2 per litre of diesel (IPCC 2006 defaults)


def load_countries(production=PRODUCTION, world_bank=WORLD_BANK) -> pd.DataFrame:
    """Economies with incomplete rural access, with their rural population without access and their crops."""
    prod = pd.read_csv(production)
    wb = pd.read_csv(world_bank)
    data = wb.merge(prod, on="iso3", how="left").fillna({"rice_t": 0.0, "maize_t": 0.0})
    data["unserved"] = data.rural_pop * (1 - data.rural_access_pct / 100)
    return data


def estimate(yields: pd.DataFrame, feeds: pd.DataFrame, t_op: float, data: pd.DataFrame | None = None,
             draws: int = N_DRAWS, seed: int = SEED):
    """Monte Carlo estimate by country, region and world, for the two tiers of demand.

    Returns the summary table (world and regions), the table by country and the samples of the electricity per
    tonne of rice husk and of corncob (kWh per tonne as received).
    """
    data = load_countries() if data is None else data
    rng = np.random.default_rng(seed)
    e_rh = electricity_draws(yields, feeds, "rice husk (pelletized)", t_op, draws, rng)
    e_cc = electricity_draws(yields, feeds, "corncob (pelletized)", t_op, draws, rng)
    rpr_r = rng.uniform(*RPR["rice"], draws)
    rpr_m = rng.uniform(*RPR["maize"], draws)
    avail = rng.uniform(*AVAILABLE, draws)
    hh = rng.uniform(*HOUSEHOLD, draws)
    sfc = rng.uniform(*DIESEL, draws)

    rice = data.rice_t.to_numpy()[:, None]
    maize = data.maize_t.to_numpy()[:, None]
    unserved = data.unserved.to_numpy()[:, None]
    energy = rice * rpr_r * avail * e_rh + maize * rpr_m * avail * e_cc            # kWh/yr, (country, draw)

    summary, by_country = [], []
    for tier, kwh_hh in TIERS.items():
        per_capita = kwh_hh / hh                                                    # kWh/person/yr
        demand = unserved * per_capita
        served = np.minimum(energy, demand)
        people = np.minimum(unserved, energy / per_capita)
        diesel = served * sfc                                                       # L/yr
        totals = {
            "electricity_TWh": energy.sum(0) / 1e9, "demand_TWh": demand.sum(0) / 1e9,
            "coverage": served.sum(0) / demand.sum(0), "people_served_M": people.sum(0) / 1e6,
            "diesel_GL": diesel.sum(0) / 1e9, "co2_Mt": diesel.sum(0) * CO2_PER_LITRE / 1e9,
            "countries_fully_covered": (served >= demand - 1e-9).sum(0),
        }
        row = {"tier": tier, "T_op": t_op, "region": "World", "unserved_M": float(unserved.sum() / 1e6)}
        for k, v in totals.items():
            row[k] = np.median(v)
            row[k + "_p05"] = np.quantile(v, 0.05)
            row[k + "_p95"] = np.quantile(v, 0.95)
        summary.append(row)
        cov = np.where(demand > 0, served / np.maximum(demand, 1e-12), np.nan)
        for i, iso in enumerate(data.iso3):
            by_country.append({"tier": tier, "T_op": t_op, "iso3": iso, "region": data.region.iloc[i],
                               "unserved_M": data.unserved.iloc[i] / 1e6,
                               "electricity_GWh": np.median(energy[i]) / 1e6,
                               "coverage": np.nanmedian(cov[i]), "coverage_p05": np.nanquantile(cov[i], 0.05),
                               "coverage_p95": np.nanquantile(cov[i], 0.95),
                               "people_served_M": np.median(people[i]) / 1e6})
        for region, idx in data.groupby("region").groups.items():
            idx = np.array(list(idx))
            d = demand[idx].sum(0)
            share = served[idx].sum(0) / np.maximum(d, 1e-12)
            summary.append({"tier": tier, "T_op": t_op, "region": region,
                            "unserved_M": float(unserved[idx].sum() / 1e6),
                            "electricity_TWh": np.median(energy[idx].sum(0)) / 1e9,
                            "coverage": np.median(share), "coverage_p05": np.quantile(share, 0.05),
                            "coverage_p95": np.quantile(share, 0.95),
                            "people_served_M": np.median(people[idx].sum(0)) / 1e6})
    return pd.DataFrame(summary), pd.DataFrame(by_country), e_rh, e_cc


if __name__ == "__main__":
    yields_path, feed_path, t_op, prefix = sys.argv[1:5]
    out, countries, e_rh, e_cc = estimate(pd.read_csv(yields_path), pd.read_csv(feed_path).set_index("treatment"),
                                          float(t_op))
    out.to_csv(prefix + "_summary.csv", index=False)
    countries.to_csv(prefix + "_countries.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"electricity per tonne at {float(t_op):.0f} C: rice husk {np.median(e_rh):.0f} "
          f"({np.quantile(e_rh, .05):.0f}-{np.quantile(e_rh, .95):.0f}) kWh/t, corncob {np.median(e_cc):.0f} "
          f"({np.quantile(e_cc, .05):.0f}-{np.quantile(e_cc, .95):.0f}) kWh/t")
    print(out.round(3).to_string(index=False))
