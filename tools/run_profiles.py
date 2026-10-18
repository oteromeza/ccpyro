"""Profiles of one cell and one variable over the six nodes (a slice of the profile stage of ccpyro.identifiability).

usage: python run_profiles.py <cell: huskp|blendp|cobp> <variable: y_t|x_t>
Writes profile_curves__<cell>__<variable>.csv and profile_intervals__<cell>__<variable>.csv to CCPYRO_RESULTS.
"""
import sys

import pandas as pd

from ccpyro import identifiability as idf
from ccpyro.model import N_NODES, RESULTS, T_NODES

cell, variable = sys.argv[1], sys.argv[2]
contexts, states, _ = idf.build_contexts()
label = next(lab for lab, short in idf.SHORT_LABEL.items() if short == cell)
context, reference = contexts[label], states[label].vector
curves, intervals = [], []
for node in range(N_NODES):
    frame = idf.profile(context, reference, variable, node)
    frame.insert(0, "treatment", label)
    curves.append(frame)
    intervals.append({"treatment": label, "variable": variable, "node": node, "T": T_NODES[node],
                      **idf.profile_interval(frame)})
    pd.concat(curves).to_csv(RESULTS / f"profile_curves__{cell}__{variable}.csv", index=False)
    pd.DataFrame(intervals).to_csv(RESULTS / f"profile_intervals__{cell}__{variable}.csv", index=False)
    print(f"profile {label} {variable} node {node}", flush=True)
