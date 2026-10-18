"""Extend each profile-likelihood interval by the ensemble witnesses.

A profile point is a constrained re-optimisation warm-started from its
neighbour. When that re-optimisation fails to reach an admissible state the
value is skipped, and a walk that fails repeatedly stops short of the true edge
of the interval, so the profile alone can only understate an interval. Every
member of the ensemble of the same criterion is an admissible converged state,
so a member whose criterion lies within the threshold of the minimum is a
witness that its value of the profiled coordinate belongs to the interval. The
reported interval is therefore the hull of the admitted profile points and the
admitted ensemble members, both referred to the lower of the two minima, and the
profile-only limits are kept alongside for comparison.
"""

from __future__ import annotations

import pandas as pd

from ccpyro.model import RESULTS

CHI2_95_1DOF = 3.8415
LEVELS = {
    "measurements": "data_misfit",
    "closures": "closure_misfit",
    "estimator": "estimator_misfit",
}


def main() -> None:
    curves = pd.read_csv(RESULTS / "profile_curves.csv")
    intervals = pd.read_csv(RESULTS / "profile_intervals.csv")
    ensembles = {level: pd.read_csv(RESULTS / f"ensemble_{level}.csv") for level in LEVELS}
    rows = []
    for _, row in intervals.iterrows():
        out = row.to_dict()
        tag = int(row["T"])
        for level, statistic in LEVELS.items():
            profile = curves[
                (curves["treatment"] == row["treatment"])
                & (curves["variable"] == row["variable"])
                & (curves["node"] == row["node"])
                & (curves["criterion"] == level)
            ]
            ensemble = ensembles[level]
            ensemble = ensemble[ensemble["treatment"] == row["treatment"]]
            floor = min(profile[level].min(), ensemble[statistic].min())
            admitted = list(profile[profile[level] <= floor + CHI2_95_1DOF]["value"])
            witnesses = ensemble[ensemble[statistic] <= floor + CHI2_95_1DOF][
                f"{row['variable']}_{tag}"
            ]
            values = admitted + list(witnesses)
            out[f"{level}_profile_lower"] = row[f"{level}_lower"]
            out[f"{level}_profile_upper"] = row[f"{level}_upper"]
            out[f"{level}_lower"] = float(min(values))
            out[f"{level}_upper"] = float(max(values))
            out[f"{level}_extended"] = bool(
                out[f"{level}_lower"] < row[f"{level}_lower"] - 1e-9
                or out[f"{level}_upper"] > row[f"{level}_upper"] + 1e-9
            )
        rows.append(out)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "profile_intervals.csv", index=False)
    extended = sum(frame[f"{level}_extended"].sum() for level in LEVELS)
    print(f"{extended} of {3 * len(frame)} intervals extended by ensemble witnesses")


if __name__ == "__main__":
    main()
