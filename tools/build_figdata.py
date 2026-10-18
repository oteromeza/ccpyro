"""Reshape the identifiability results into the wide tables the figures read."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccpyro.model import FIGDATA, RESULTS, T_NODES, SHORT

LEVELS = ("measurements", "closures", "estimator")


def profile_bands() -> None:
    """Profile-likelihood interval of each condensate ratio at every node."""
    frame = pd.read_csv(RESULTS / "profile_intervals.csv")
    for variable in ("y_t", "x_t"):
        wide = pd.DataFrame({"T": T_NODES})
        for treatment, code in SHORT.items():
            subset = frame[
                (frame["treatment"] == treatment) & (frame["variable"] == variable)
            ].sort_values("T")
            if subset.empty:
                continue
            for level in LEVELS:
                wide[f"{level}_lo_{code}"] = subset[f"{level}_lower"].to_numpy()
                wide[f"{level}_hi_{code}"] = subset[f"{level}_upper"].to_numpy()
            # The line drawn over the bars is the reported estimate, which is
            # the best state of any search and not necessarily the reference
            # the profile walk started from.
            reported = pd.read_csv(RESULTS / "product_yields.csv")
            reported = reported[reported["treatment"] == treatment].sort_values("T")
            wide[f"best_{code}"] = reported[variable].to_numpy()
        wide.to_csv(FIGDATA / f"profile_{variable}.csv", index=False)


def profile_curves() -> None:
    """Rise of each criterion along the coordinate, at the last node."""
    frame = pd.read_csv(RESULTS / "profile_curves.csv")
    node = int(T_NODES.size - 1)
    for variable in ("y_t", "x_t"):
        pieces = []
        for treatment, code in SHORT.items():
            subset = frame[
                (frame["treatment"] == treatment)
                & (frame["variable"] == variable)
                & (frame["node"] == node)
            ]
            for level in LEVELS:
                piece = subset[subset["criterion"] == level][["value", level]].copy()
                if piece.empty:
                    continue
                piece = piece.sort_values("value")
                piece[level] = piece[level] - piece[level].min()
                piece = piece.rename(columns={"value": f"v_{level}_{code}",
                                              level: f"d_{level}_{code}"})
                pieces.append(piece.reset_index(drop=True))
        if pieces:
            pd.concat(pieces, axis=1).to_csv(
                FIGDATA / f"profile_curve_{variable}.csv", index=False
            )


def envelope_band() -> None:
    """Envelope of the sampled admissible set, level by level."""
    for level in LEVELS:
        path = RESULTS / f"envelope_{level}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        wide = pd.DataFrame({"T": T_NODES})
        for treatment, code in SHORT.items():
            subset = frame[frame["treatment"] == treatment].sort_values("T")
            if subset.empty:
                continue
            for name in ("y_t", "x_t", "w_tar", "w_gas", "w_liquid", "water_in_liquid"):
                wide[f"{name}_lo_{code}"] = subset[f"{name}_min"].to_numpy()
                wide[f"{name}_hi_{code}"] = subset[f"{name}_max"].to_numpy()
                wide[f"{name}_md_{code}"] = subset[f"{name}_median"].to_numpy()
        wide.to_csv(FIGDATA / f"envelope_{level}.csv", index=False)


def spectrum() -> None:
    """Eigenvalue spectra of the three criteria, side by side."""
    frame = pd.read_csv(RESULTS / "identifiability_spectrum.csv")
    wide = None
    for treatment, code in SHORT.items():
        for level in LEVELS:
            subset = frame[
                (frame["treatment"] == treatment) & (frame["criterion"] == level)
            ].sort_values("index")
            if subset.empty:
                continue
            column = np.maximum(subset["relative"].to_numpy(), 1e-18)
            if wide is None:
                wide = pd.DataFrame({"index": subset["index"].to_numpy()})
            wide[f"{level}_{code}"] = column
    if wide is not None:
        wide.to_csv(FIGDATA / "identifiability_spectrum.csv", index=False)


def main() -> None:
    FIGDATA.mkdir(exist_ok=True)
    spectrum()
    for builder in (profile_bands, profile_curves, envelope_band):
        try:
            builder()
        except FileNotFoundError as error:
            print(f"skipped {builder.__name__}: {error}", flush=True)
    print("figure data written", flush=True)


if __name__ == "__main__":
    main()
