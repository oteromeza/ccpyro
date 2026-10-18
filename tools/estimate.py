"""Point estimate of every cell: two sequential passes and independent admissible starts (Algorithm 1).

usage: python tools/estimate.py [--random-starts 40]
The best decision vector of each cell is cached in results/best_vectors.npz.
"""
import argparse

import numpy as np

from ccpyro import model as mp

parser = argparse.ArgumentParser()
parser.add_argument("--random-starts", type=int, default=40)
arguments = parser.parse_args()
built = mp.build_cases(passes=2, random_starts=arguments.random_starts)
for case in mp.CASES:
    label = mp.case_name(case)
    state = built["states"][label]
    print(f"{label:24s} E = {state.objective:.5f}  w_tar = {np.round(state.w_tar, 3)}  "
          f"x_t = {np.round(state.x_t, 3)}  y_t = {np.round(state.y_t, 3)}", flush=True)
