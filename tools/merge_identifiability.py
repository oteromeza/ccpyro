"""Join the per-cell result files of the identifiability stage.

The spectra, ensembles and envelopes are written by ``python -m ccpyro.identifiability --cell <cell>`` with the
cell as suffix, and the profiles by ``tools/run_profiles.py`` with the cell and the variable as suffix.
"""

from __future__ import annotations

import pandas as pd

from ccpyro.model import RESULTS

CELLS = ("huskp", "blendp", "cobp")
VARIABLES = ("y_t", "x_t")
STEMS = (
    "identifiability_spectrum",
    "identifiability_vectors",
    "ensemble_estimator",
    "ensemble_closures",
    "ensemble_measurements",
    "envelope_estimator",
    "envelope_closures",
    "envelope_measurements",
)


def main() -> None:
    for stem in STEMS:
        parts = [RESULTS / f"{stem}__{cell}.csv" for cell in CELLS]
        present = [path for path in parts if path.exists()]
        if len(present) != len(parts):
            print(f"{stem}: {len(present)} of {len(parts)} cells present, skipped")
            continue
        pd.concat([pd.read_csv(path) for path in present]).to_csv(RESULTS / f"{stem}.csv", index=False)
        print(f"{stem}: merged")
    for stem in ("profile_curves", "profile_intervals"):
        parts = [RESULTS / f"{stem}__{cell}__{variable}.csv" for cell in CELLS for variable in VARIABLES]
        present = [path for path in parts if path.exists()]
        if len(present) != len(parts):
            print(f"{stem}: {len(present)} of {len(parts)} slices present, skipped")
            continue
        pd.concat([pd.read_csv(path) for path in present]).to_csv(RESULTS / f"{stem}.csv", index=False)
        print(f"{stem}: merged")


if __name__ == "__main__":
    main()
