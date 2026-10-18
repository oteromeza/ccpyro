"""Identifiability of the constrained pyrolysis estimate.

Three instruments are provided, all of them operating on the model defined in
``model_pyrolysis`` and none of them requiring a new measurement.

``multistart``
    Solves the estimation problem from many admissible starting points. A
    non-convex programme of fifty-two unknowns, most of whose directions are
    flat, can return a different state for every start, so a single start
    demonstrates nothing about the representativeness of the reported estimate.

``admissible_ensemble``
    Minimises the measurement misfit alone, subject to exact conservation and to
    the admissibility constraints, from the same family of starting points. The
    resulting sample populates the set of states that the measurements and the
    physics jointly admit, without the empirical closures, the anchors or the
    prior. Its envelope is the object the manuscript reports.

``profile``
    Profile likelihood in the sense of Raue et al. One component of the decision
    vector is fixed at a sequence of values and the remaining fifty-one are
    re-optimised, which traces the increase of the misfit along that coordinate
    and delivers a confidence interval that is valid without a linearisation.
    Both the full objective and the measurement misfit alone are profiled, and
    the difference between the two intervals is the contribution of the
    structural terms.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import (
    CLOSURE_BLOCKS,
    DATA_BLOCKS,
    IDX,
    M,
    N_FREE,
    N_NODES,
    OBJECTIVE_SCALE,
    RESULTS,
    T_NODES,
    build_cases,
    ModelContext,
    Solution,
    constraints,
    closure_objective,
    data_objective,
    level_misfit,
    objective,
    objective_terms,
    performance_indices,
    restore_feasibility,
    unpack,
)

# Threshold of the likelihood-ratio test for one degree of freedom at the
# 95 per cent level. The misfits of Eqs. (49) to (51) are sums of squared
# standardised residuals, so the increase of each sum along a profile is
# compared with it directly.
CHI2_95_1DOF = 3.8415
PROFILE_VARIABLES = ("y_t", "x_t")
PROFILE_POINTS = int(os.environ.get("CCPYRO_PROFILE_POINTS", 13))


def random_admissible_start(context: ModelContext, rng: np.random.Generator) -> np.ndarray:
    """Draw one starting point from the admissible domain.

    The draw deliberately ignores the anchors and the prior, so that the sample
    of starting points is not concentrated where the reported estimate lies. The
    carbonisation exponents, the end-of-range composition of both condensed
    phases, the shape of the two condensate trajectories and the split of the
    volatile carbon between gas and condensate are all randomised; the result is
    then projected onto the feasible set.
    """
    theta = context.theta
    x0, y0 = context.x0, context.y0

    a_H = float(rng.uniform(0.2, 2.2))
    a_O = float(rng.uniform(0.2, 2.6))
    x_c = x0 * theta**a_H
    y_c = y0 * theta**a_O
    n_c = theta * context.molar_mass / (M["C"] + M["H"] * x_c + M["O"] * y_c)

    # Monotone non-increasing condensate trajectories between a randomised start
    # and a randomised end, with a randomised curvature.
    shape = float(rng.uniform(0.4, 3.0))
    ramp = np.linspace(0.0, 1.0, N_NODES) ** shape
    x_start = float(rng.uniform(0.75 * x0, x0))
    x_end = float(rng.uniform(0.45, max(0.46, x_start)))
    y_start = float(rng.uniform(0.55 * y0, y0))
    y_end = float(rng.uniform(0.005, max(0.006, 0.9 * y_start)))
    x_t = x_start + (x_end - x_start) * ramp
    y_t = y_start + (y_end - y_start) * ramp

    # The condensate cannot be more condensed than the solid.
    x_t = np.minimum(np.maximum(x_t, x_c * 1.01), x0)
    y_t = np.minimum(np.maximum(y_t, y_c * 1.01), y0)
    x_t = np.minimum.accumulate(x_t)
    y_t = np.minimum.accumulate(y_t)

    block = np.zeros((N_NODES, N_FREE))
    block[:, IDX["n_c"]] = n_c
    block[:, IDX["x_c"]] = x_c
    block[:, IDX["y_c"]] = y_c
    block[:, IDX["x_t"]] = x_t
    block[:, IDX["y_t"]] = y_t

    volatile_carbon = np.maximum(1.0 - n_c, 1e-6)
    fractions = context.composition[:, :3] / context.composition[:, :3].sum(
        axis=1, keepdims=True
    )
    share = np.sort(rng.uniform(0.05, 0.95, size=N_NODES))
    gas = share[:, None] * volatile_carbon[:, None] * fractions
    for column in range(3):
        gas[:, column] = np.maximum.accumulate(gas[:, column])
    block[:, IDX["n_CO"] : IDX["n_CH4"] + 1] = gas

    vector = block.reshape(-1)
    if context.calibrate:
        parameters = np.array(
            [
                float(rng.uniform(0.05, 2.0)),
                float(rng.uniform(0.10, 0.95)),
                a_H,
                a_O,
            ]
        )
        vector = np.concatenate([vector, parameters])
    vector = np.clip(vector, context.lower + 1e-9, context.upper - 1e-9)
    return restore_feasibility(vector, context)


def minimise(
    context: ModelContext,
    start: np.ndarray,
    criterion=objective,
    max_iterations: int = 400,
    tolerance: float = 1e-10,
    extra_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Sequential quadratic programming against ``criterion`` from one start."""
    lower, upper = context.lower, context.upper
    if extra_bounds is not None:
        lower, upper = extra_bounds
    vector = np.clip(start, lower + 1e-12, upper - 1e-12)
    result = minimize(
        criterion,
        vector,
        args=(context,),
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[{"type": "ineq", "fun": constraints, "args": (context,)}],
        options={"maxiter": max_iterations, "ftol": tolerance},
    )
    feasibility = float(np.min(constraints(result.x, context)))
    return result.x, float(result.fun), feasibility


def _record(vector: np.ndarray, context: ModelContext, index: int) -> dict:
    state = unpack(vector, context)
    indices = performance_indices(state, context)
    terms = objective_terms(vector, context)
    row = {
        "start": index,
        "objective": sum(terms.values()) / OBJECTIVE_SCALE,
        "estimator_misfit": sum(terms.values()),
        "closure_misfit": sum(terms[key] for key in CLOSURE_BLOCKS),
        "data_misfit": sum(terms[key] for key in DATA_BLOCKS),
        "feasibility": float(np.min(constraints(vector, context))),
        "CGE_800": indices["CGE"][-1],
        "A": state.parameters[1],
        "kappa": state.parameters[0],
        "a_H": state.parameters[2],
        "a_O": state.parameters[3],
    }
    for k, temperature in enumerate(T_NODES):
        tag = int(temperature)
        row[f"y_t_{tag}"] = state.y_t[k]
        row[f"x_t_{tag}"] = state.x_t[k]
        row[f"y_c_{tag}"] = state.y_c[k]
        row[f"x_c_{tag}"] = state.x_c[k]
        row[f"w_liquid_{tag}"] = state.w_liquid[k]
        row[f"w_tar_{tag}"] = state.w_tar[k]
        row[f"w_water_{tag}"] = state.w_water[k]
        row[f"w_gas_{tag}"] = state.w_gas[k]
        row[f"water_in_liquid_{tag}"] = state.w_water[k] / max(state.w_liquid[k], 1e-12)
    return row


def multistart(
    context: ModelContext,
    draws: int,
    rng: np.random.Generator,
    criterion=objective,
    reference: np.ndarray | None = None,
) -> pd.DataFrame:
    """Solve from ``draws`` independent admissible starting points."""
    rows = []
    if reference is not None:
        rows.append(_record(reference, context, -1))
    for index in range(draws):
        start = random_admissible_start(context, rng)
        vector, _, feasibility = minimise(context, start, criterion=criterion)
        if feasibility < -1e-5:
            continue
        rows.append(_record(vector, context, index))
    return pd.DataFrame(rows)


def envelope(frame: pd.DataFrame, threshold: float, statistic: str) -> pd.DataFrame:
    """Envelope of the trajectories whose misfit lies within ``threshold``."""
    best = frame[statistic].min()
    kept = frame[frame[statistic] <= best + threshold]
    rows = []
    for temperature in T_NODES:
        tag = int(temperature)
        row = {"T": temperature, "members": len(kept)}
        for name in (
            "y_t",
            "x_t",
            "y_c",
            "x_c",
            "w_liquid",
            "w_tar",
            "w_water",
            "w_gas",
            "water_in_liquid",
        ):
            values = kept[f"{name}_{tag}"].to_numpy(float)
            row[f"{name}_min"] = values.min()
            row[f"{name}_max"] = values.max()
            row[f"{name}_median"] = float(np.median(values))
        rows.append(row)
    return pd.DataFrame(rows)


def profile(
    context: ModelContext,
    reference: np.ndarray,
    variable: str,
    node: int,
    points: int = PROFILE_POINTS,
) -> pd.DataFrame:
    """Trace the misfit along one coordinate of the decision vector.

    The coordinate is fixed by collapsing its box bounds, the remaining
    coordinates are re-optimised, and the walk proceeds outwards from the
    reference in both directions so that every solve is warm-started from its
    neighbour. Both criteria are profiled: the full objective, which is what the
    estimate minimises, and the measurement misfit alone, which is what the data
    can distinguish.
    """
    position = node * N_FREE + IDX[variable]
    low, high = context.lower[position], context.upper[position]
    anchor = float(reference[position])
    # A uniform grid resolves the flat part of the profile but not its rise near
    # the optimum, so a geometric refinement is added around the reference.
    span = high - low
    refinement = anchor + span * np.array(
        [-0.16, -0.08, -0.04, -0.02, -0.01, 0.01, 0.02, 0.04, 0.08, 0.16]
    )
    grid = np.unique(
        np.clip(
            np.concatenate([np.linspace(low, high, points), refinement, [anchor]]),
            low,
            high,
        )
    )

    rows = []
    for criterion, tag in (
        (objective, "estimator"),
        (closure_objective, "closures"),
        (data_objective, "measurements"),
    ):
        for direction in (1, -1):
            values = grid[grid >= anchor] if direction == 1 else grid[grid <= anchor][::-1]
            warm = reference.copy()
            for value in values:
                lower, upper = context.lower.copy(), context.upper.copy()
                lower[position] = upper[position] = value
                warm[position] = value
                vector, _, feasibility = minimise(
                    context,
                    warm,
                    criterion=criterion,
                    extra_bounds=(lower, upper),
                    max_iterations=250,
                )
                if feasibility < -1e-5:
                    continue
                warm = vector
                terms = objective_terms(vector, context)
                rows.append(
                    {
                        "criterion": tag,
                        "variable": variable,
                        "node": node,
                        "T": T_NODES[node],
                        "value": value,
                        "estimator": sum(terms.values()),
                        "closures": sum(terms[key] for key in CLOSURE_BLOCKS),
                        "measurements": sum(terms[key] for key in DATA_BLOCKS),
                        "feasibility": feasibility,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["criterion", "value"])
    return frame.sort_values(["criterion", "value"]).reset_index(drop=True)


def profile_interval(frame: pd.DataFrame) -> dict:
    """Confidence interval of one profile at each level of evidence.

    Each level is profiled against its own criterion, so the interval reported
    for a level is the set of values that the evidence of that level cannot
    distinguish from its own best fit at the 95 per cent level.
    """
    out = {}
    for tag in ("measurements", "closures", "estimator"):
        subset = frame[frame["criterion"] == tag]
        if subset.empty:
            continue
        best = subset[tag].min()
        inside = subset[subset[tag] <= best + CHI2_95_1DOF]["value"]
        out[f"{tag}_lower"] = float(inside.min())
        out[f"{tag}_upper"] = float(inside.max())
        out[f"{tag}_best"] = float(subset.loc[subset[tag].idxmin(), "value"])
        out[f"{tag}_min"] = float(best)
    return out


def hessian(criterion, vector: np.ndarray, context: ModelContext) -> np.ndarray:
    span = np.maximum(context.upper - context.lower, 1e-6)
    step = 1e-4 * span
    size = vector.size
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            shift_i, shift_j = np.zeros(size), np.zeros(size)
            shift_i[i], shift_j[j] = step[i], step[j]
            plus = criterion(vector + shift_i + shift_j, context)
            minus = criterion(vector - shift_i - shift_j, context)
            cross = criterion(vector + shift_i - shift_j, context)
            anti = criterion(vector - shift_i + shift_j, context)
            matrix[i, j] = matrix[j, i] = (plus + minus - cross - anti) / (
                4.0 * step[i] * step[j]
            )
    return matrix * np.outer(span, span)


def _labels(context: ModelContext) -> list[str]:
    names = []
    for k, temperature in enumerate(T_NODES):
        for variable in ("n_c", "x_c", "y_c", "x_t", "y_t", "n_CO", "n_CO2", "n_CH4"):
            names.append(f"{variable}({int(temperature)})")
    if context.calibrate:
        names += ["kappa_W", "A", "a_H", "a_O"]
    return names


def spectra(context: ModelContext, reference: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Eigenvalue spectra and leading eigenvectors of both criteria."""
    labels = _labels(context)
    rows, vectors = [], []
    criteria = (
        (lambda v, c: level_misfit(v, c, "measurements"), "measurements"),
        (lambda v, c: level_misfit(v, c, "closures"), "closures"),
        (lambda v, c: level_misfit(v, c, "estimator"), "estimator"),
    )
    for criterion, tag in criteria:
        matrix = hessian(criterion, reference, context)
        values, directions = np.linalg.eigh(0.5 * (matrix + matrix.T))
        order = np.argsort(values)[::-1]
        values, directions = values[order], directions[:, order]
        largest = values[0]
        for index, value in enumerate(values):
            rows.append(
                {
                    "criterion": tag,
                    "index": index + 1,
                    "eigenvalue": value,
                    "relative": value / largest,
                }
            )
        for index in range(6):
            weights = directions[:, index] ** 2
            leading = np.argsort(weights)[::-1][:5]
            vectors.append(
                {
                    "criterion": tag,
                    "index": index + 1,
                    "eigenvalue": values[index],
                    "relative": values[index] / largest,
                    "composition": "; ".join(
                        f"{labels[i]} {directions[i, index]:+.2f}" for i in leading
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(vectors)


def build_contexts() -> tuple[dict[str, ModelContext], dict[str, Solution], np.ndarray]:
    """Build the three pelletised cells through the model's own driver."""
    built = build_cases(random_starts=0)
    return built["contexts"], built["states"], built["relative_sd"]


SHORT_LABEL = {
    "rice husk (pelletized)": "huskp",
    "blend (pelletized)": "blendp",
    "corncob (pelletized)": "cobp",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starts", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--stage", default="all")
    parser.add_argument(
        "--criterion",
        default="all",
        help="estimator, closures or measurements, to run the multistart of one criterion alone",
    )
    parser.add_argument(
        "--cell",
        default="all",
        help="short label of one cell (huskp, blendp, cobp) to run it alone; the "
        "result files then carry the label as a suffix and merge_identifiability.py "
        "joins them",
    )
    arguments = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    contexts, states, _ = build_contexts()
    labels = list(contexts)
    if arguments.cell != "all":
        labels = [label for label in labels if SHORT_LABEL[label] == arguments.cell]
    suffix = "" if arguments.cell == "all" else f"__{arguments.cell}"

    def generator(label: str) -> np.random.Generator:
        # One stream per cell, so that a cell draws the same starting points
        # whether it is run alone or together with the others.
        return np.random.default_rng([arguments.seed, list(SHORT_LABEL).index(label)])

    if arguments.stage in ("all", "spectra"):
        frames, vectors = [], []
        for treatment in labels:
            context = contexts[treatment]
            spectrum, directions = spectra(context, states[treatment].vector)
            spectrum.insert(0, "treatment", treatment)
            directions.insert(0, "treatment", treatment)
            frames.append(spectrum)
            vectors.append(directions)
        pd.concat(frames).to_csv(RESULTS / f"identifiability_spectrum{suffix}.csv", index=False)
        pd.concat(vectors).to_csv(RESULTS / f"identifiability_vectors{suffix}.csv", index=False)
        print("spectra written", flush=True)

    if arguments.stage in ("all", "multistart"):
        for tag, criterion, statistic in (
            ("estimator", objective, "estimator_misfit"),
            ("closures", closure_objective, "closure_misfit"),
            ("measurements", data_objective, "data_misfit"),
        ):
            if arguments.criterion not in ("all", tag):
                continue
            ensembles, envelopes = [], []
            for treatment in labels:
                context = contexts[treatment]
                frame = multistart(
                    context,
                    arguments.starts,
                    generator(treatment),
                    criterion=criterion,
                    reference=states[treatment].vector,
                )
                frame.insert(0, "treatment", treatment)
                ensembles.append(frame)
                band = envelope(frame, CHI2_95_1DOF, statistic)
                band.insert(0, "treatment", treatment)
                envelopes.append(band)
                print(f"{tag} {treatment}: {len(frame)} solutions", flush=True)
                pd.concat(ensembles).to_csv(RESULTS / f"ensemble_{tag}{suffix}.csv", index=False)
                pd.concat(envelopes).to_csv(RESULTS / f"envelope_{tag}{suffix}.csv", index=False)

    if arguments.stage in ("all", "profiles"):
        # Each profile is written as soon as it is computed, because the stage
        # takes hours and a partial run is still usable.
        curves, intervals = [], []
        for treatment in labels:
            context = contexts[treatment]
            reference = states[treatment].vector
            for variable in PROFILE_VARIABLES:
                for node in range(N_NODES):
                    frame = profile(context, reference, variable, node)
                    frame.insert(0, "treatment", treatment)
                    curves.append(frame)
                    intervals.append(
                        {
                            "treatment": treatment,
                            "variable": variable,
                            "node": node,
                            "T": T_NODES[node],
                            **profile_interval(frame),
                        }
                    )
                    pd.concat(curves).to_csv(
                        RESULTS / f"profile_curves{suffix}.csv", index=False
                    )
                    pd.DataFrame(intervals).to_csv(
                        RESULTS / f"profile_intervals{suffix}.csv", index=False
                    )
                    print(f"profile {treatment} {variable} node {node}", flush=True)


if __name__ == "__main__":
    main()
