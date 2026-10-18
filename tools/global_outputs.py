"""Energy balance of the unit and global estimate from the results of the framework.

usage: python tools/global_outputs.py
Writes results/unit_energy.csv and results/global_<T>_summary.csv, results/global_<T>_countries.csv for T = 500, 800.
"""
import runpy
import sys

import pandas as pd

from ccpyro import global_potential as gp
from ccpyro.model import RESULTS

yields_path, feed_path = RESULTS / "product_yields.csv", RESULTS / "feedstock_characterisation.csv"
sys.argv = ["energy", str(yields_path), str(feed_path), str(RESULTS / "unit_energy.csv")]
runpy.run_module("ccpyro.energy", run_name="__main__")
yields, feeds = pd.read_csv(yields_path), pd.read_csv(feed_path).set_index("treatment")
for t_op in (500.0, 800.0):
    summary, countries, e_rh, e_cc = gp.estimate(yields, feeds, t_op)
    summary.to_csv(RESULTS / f"global_{t_op:.0f}_summary.csv", index=False)
    countries.to_csv(RESULTS / f"global_{t_op:.0f}_countries.csv", index=False)
    print(summary[summary.region == "World"].round(3).to_string(index=False), flush=True)
