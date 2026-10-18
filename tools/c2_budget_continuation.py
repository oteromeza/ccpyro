"""Recompute the omitted-species budget with a continuation in the lump ratio.

The re-solve at a fixed lump abundance occasionally lands in a poor basin when
it is warm-started only from the four-species state, because carrying the lump
displaces carbon between the gas and the condensate over the whole grid at once.
Walking the three scenarios in order of increasing abundance, and warm-starting
each from the accepted state of the previous one, removes those failures; a
candidate is accepted only if its objective is comparable to the four-species
optimum, and otherwise the solve is repeated from the reference with more
starts and the better of the two is kept.
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


def record(name, scenario, state, context):
    indices = performance_indices(state, context)
    return {
        "treatment": name,
        "scenario": scenario,
        "objective": state.objective,
        "w_tar_400": state.w_tar[1],
        "w_tar_800": state.w_tar[-1],
        "w_gas_800": state.w_gas[-1],
        "w_water_800": state.w_water[-1],
        "y_t_800": state.y_t[-1],
        "x_t_800": state.x_t[-1],
        "LHV_500": indices["LHV_gas"][2],
        "LHV_800": indices["LHV_gas"][-1],
        "CGE_800": indices["CGE"][-1],
    }


def main() -> None:
    built = build_cases()
    contexts, states = built["contexts"], built["states"]
    rows = []
    for name, context in contexts.items():
        reference = states[name]
        rows.append(record(name, "four species", reference, context))
        previous = reference.vector
        for scenario in ORDER:
            trial = replace(context, c2_to_methane=C2_TO_METHANE[scenario])
            state = solve(trial, start=[previous, reference.vector], restarts=3)
            if state.objective > 5.0 * reference.objective + 1.0:
                retry = solve(trial, start=reference.vector, restarts=8)
                if retry.objective < state.objective:
                    state = retry
            if state.objective <= 5.0 * reference.objective + 1.0:
                previous = state.vector
            rows.append(record(name, scenario, state, context))
            print(f"{name:24s} {scenario:8s} E={state.objective:10.4f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "c2_budget.csv", index=False)
    print("C2 budget rewritten", flush=True)


if __name__ == "__main__":
    main()
