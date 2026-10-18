"""Retry the C2 re-solves that the continuation did not bring into a good basin.

A scenario is treated as failed when its objective exceeds five times that of
the four-species optimum of the same cell. The retry draws its starting points
from every accepted state of the same scenario in the other cells, on the
grounds that neighbouring cells of the design occupy neighbouring basins.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ccpyro.model import (
    C2_TO_METHANE,
    RESULTS,
    build_cases,
    performance_indices,
    solve,
)

ORDER = ("low", "central", "high")


def main() -> None:
    frame = pd.read_csv(RESULTS / "c2_budget.csv")
    reference = frame[frame["scenario"] == "four species"].set_index("treatment")
    failed = [
        (row["treatment"], row["scenario"])
        for _, row in frame.iterrows()
        if row["scenario"] != "four species"
        and row["objective"] > 5.0 * reference.loc[row["treatment"], "objective"] + 1.0
    ]
    if not failed:
        print("no failures to retry", flush=True)
        return
    print("retrying:", failed, flush=True)

    built = build_cases()
    contexts, states = built["contexts"], built["states"]

    # Solve every cell at every scenario once, so that a failed cell can be
    # started from the accepted state of a neighbouring one.
    pool = {scenario: [] for scenario in ORDER}
    for name, context in contexts.items():
        previous = states[name].vector
        for scenario in ORDER:
            trial = replace(context, c2_to_methane=C2_TO_METHANE[scenario])
            state = solve(trial, start=[previous, states[name].vector], restarts=2)
            if state.objective <= 5.0 * states[name].objective + 1.0:
                pool[scenario].append(state.vector)
                previous = state.vector

    for name, scenario in failed:
        context = contexts[name]
        trial = replace(context, c2_to_methane=C2_TO_METHANE[scenario])
        starts = pool[scenario] + [states[name].vector]
        state = solve(trial, start=starts, restarts=4)
        indices = performance_indices(state, context)
        mask = (frame["treatment"] == name) & (frame["scenario"] == scenario)
        before = frame.loc[mask, "objective"].iloc[0]
        if state.objective >= before:
            print(f"{name} {scenario}: no improvement ({state.objective:.4f})", flush=True)
            continue
        frame.loc[mask, ["objective", "w_tar_400", "w_tar_800", "w_gas_800",
                         "w_water_800", "y_t_800", "x_t_800", "LHV_500",
                         "LHV_800", "CGE_800"]] = [
            state.objective, state.w_tar[1], state.w_tar[-1], state.w_gas[-1],
            state.w_water[-1], state.y_t[-1], state.x_t[-1],
            indices["LHV_gas"][2], indices["LHV_gas"][-1], indices["CGE"][-1],
        ]
        print(f"{name} {scenario}: {before:.4f} -> {state.objective:.4f}", flush=True)

    frame.to_csv(RESULTS / "c2_budget.csv", index=False)
    print("C2 budget repaired", flush=True)


if __name__ == "__main__":
    main()
