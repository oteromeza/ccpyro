"""Replay the best draw of a cell's estimator ensemble and cache its state.

The ensemble of identifiability.py records the summary of every converged
start but not its decision vector. The draw is reproduced from the cell's own
random stream, re-minimised with the same criterion, compared with a further solve
by the model's solver, and the better of the two is stored in results/best_vectors.npz when it lowers the objective, so
that the reported estimate is the best state any search found.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ccpyro.identifiability import SHORT_LABEL, minimise, random_admissible_start
from ccpyro.model import RESULTS, build_cases, objective, solve, state_from_vector, store_best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    arguments = parser.parse_args()

    label = next(k for k, v in SHORT_LABEL.items() if v == arguments.cell)
    ensemble = pd.read_csv(RESULTS / f"ensemble_estimator__{arguments.cell}.csv")
    best = ensemble[ensemble["start"] >= 0].sort_values("estimator_misfit").iloc[0]
    index = int(best["start"])

    built = build_cases(random_starts=0)
    context, reference = built["contexts"][label], built["states"][label]
    rng = np.random.default_rng([arguments.seed, list(SHORT_LABEL).index(label)])
    for _ in range(index):
        random_admissible_start(context, rng)
    start = random_admissible_start(context, rng)
    vector, value, feasibility = minimise(context, start, criterion=objective)
    replayed = state_from_vector(vector, context)
    polished = solve(context, start=vector, restarts=1)
    candidates = [c for c in (replayed, polished) if c is not None]
    chosen = min(candidates, key=lambda c: c.objective)
    print(f"{label}: reference E={reference.objective:.6f}, "
          f"replayed draw {index} E={value:.6f} (ensemble {best['objective']:.6f}), "
          f"feasibility {feasibility:.1e}, polished E={polished.objective:.6f}, "
          f"chosen E={chosen.objective:.6f}", flush=True)
    if chosen.objective < reference.objective:
        store_best({label: chosen})
        print("cached as the reported estimate", flush=True)
    else:
        print("reference retained", flush=True)


if __name__ == "__main__":
    main()
