"""CRECK benchmark: node table, residual-mass curve, characterisation, comparison and figure data.

usage: python tools/creck_outputs.py
Reads results/product_yields.csv for the comparison and writes creck_*.csv to results/ and figdata/.
"""
import json

import numpy as np
import pandas as pd

from ccpyro import creck as C
from ccpyro.model import FIGDATA, RESULTS, load_feedstocks

RESULTS.mkdir(parents=True, exist_ok=True)
FIGDATA.mkdir(parents=True, exist_ok=True)
yields = pd.read_csv(RESULTS / "product_yields.csv")
# main benchmark: reference species from the ultimate analysis; check: from the measured biochemical composition
for basis, suffix in (("ultimate", ""), ("biochemical", "_biochemical")):
    nodes, curves, characterisation = C.run_benchmark(basis=basis)
    nodes.to_csv(RESULTS / f"creck_nodes{suffix}.csv", index=False)
    curves.to_csv(RESULTS / f"creck_thermogram{suffix}.csv", index=False)
    (RESULTS / f"creck_characterisation{suffix}.json").write_text(json.dumps(characterisation, indent=2))
    C.compare_with_framework(nodes, yields).to_csv(RESULTS / f"creck_vs_framework{suffix}.csv", index=False)
    print(f"--- {basis}")
    nodes = pd.read_csv(RESULTS / "creck_nodes.csv")

# fine-grid curves and stage-wise gas for the figure of the benchmark, one column per feedstock
CODE = {"rice husk": "huskp", "blend": "blendp", "corncob": "cobp"}
mech = C.load_mechanism()
programme = C.Programme()
grid = np.round(np.arange(200.0, 840.0 + 1e-9, 1.0), 6)
wide, stage = None, None
for name, feed in load_feedstocks().items():
    code = CODE[name]
    c = feed.daf_composition
    w = np.array([c["C"], c["H"], c["O"]]) / (c["C"] + c["H"] + c["O"])
    ch = C.characterise_admissible(*w)
    n0 = C.initial_moles(mech, {k: max(v, 0.0) for k, v in ch["species"].items()}, hemicellulose="XYGR")
    carbon0, mass0 = float(n0 @ mech.atoms[:, 0]), float(n0 @ mech.mw)
    states = C.integrate(mech, n0, programme, grid)
    solid_ref = C.lumps(mech, states[0])["mass"]["solid"]
    rows = []
    for k, T in enumerate(grid):
        L = C.lumps(mech, states[k])
        te = L["tar_elements"]
        gas = np.array([L["gas"][g] for g in ("CO", "CO2", "CH4", "H2")]) / carbon0
        rows.append({"T": T, f"theta_{code}": L["mass"]["solid"] / solid_ref, f"w_tar_{code}": L["mass"]["tar"] / mass0,
                     f"x_t_{code}": te[1] / te[0] if te[0] > 0 else np.nan,
                     f"y_t_{code}": te[2] / te[0] if te[0] > 0 else np.nan,
                     "n_CO": gas[0], "n_CO2": gas[1], "n_CH4": gas[2], "n_H2": gas[3]})
    d = pd.DataFrame(rows)
    previous, released = np.zeros(4), []
    for T in (300.0, 400.0, 500.0, 600.0, 700.0, 800.0):
        g = d.loc[np.isclose(d["T"], T), ["n_CO", "n_CO2", "n_CH4", "n_H2"]].to_numpy(float)[0]
        rel = g - previous
        previous = g
        released.append({"T": T, **{f"phi_{s}_{code}": v for s, v in zip(("CO", "CO2", "CH4", "H2"), rel / rel.sum())}})
    d = d.drop(columns=["n_CO", "n_CO2", "n_CH4", "n_H2"])
    wide = d if wide is None else wide.merge(d, on="T")
    s = pd.DataFrame(released)
    stage = s if stage is None else stage.merge(s, on="T")
wide[np.isclose(wide["T"] % 5, 0)].to_csv(FIGDATA / "creck_curves.csv", index=False)
stage.to_csv(FIGDATA / "creck_stage_gas.csv", index=False)
print(nodes.round(4).to_string(index=False))
