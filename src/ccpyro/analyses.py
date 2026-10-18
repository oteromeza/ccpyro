"""Error budget of the gas-phase measurement basis.

Two features of the chromatographic method propagate into every quantity the
framework reports, and neither has an independent dispersion that a routine
Monte Carlo can draw.

Carbon dioxide is obtained by difference. After the oxygen and nitrogen of the
admitted air are removed and the record is renormalised over the four remaining
species, the by-difference construction survives as an identity, so the error of
carbon dioxide is minus the sum of the errors of the three measured channels. A
propagation that perturbs all four channels independently therefore both
misstates the dispersion of carbon dioxide and destroys its correlation with the
rest of the record. The first analysis quantifies what that costs.

Hydrocarbons of two and three carbon atoms are not quantified at all. They carry
carbon and hydrogen that the balances assign elsewhere, they dilute the four
species that are measured, and they carry a volumetric heating value between
1.6 and 2.5 times that of methane. The second analysis re-solves the model with a
literature lump of those species included in the balances and reports the bias in
every quantity the manuscript states.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .model import (
    C2_CARBON,
    C2_TO_METHANE,
    DATA,
    GAS_SPECIES,
    LHV_C2,
    LHV_VOLUMETRIC,
    N_MONTE_CARLO,
    RESULTS,
    T_NODES,
    CASES,
    build_cases,
    case_name,
    injection_sequence,
    ModelContext,
    oxygen_bound,
    performance_indices,
    propagate,
    read_thermogram,
    solve,
    collect,
)

RANDOM_SEED = 20260726
TRACK = ("w_char", "w_tar", "w_water", "w_gas", "w_liquid", "CGE", "LHV_gas", "x_t", "y_t")


def build() -> tuple[dict[str, ModelContext], dict, np.ndarray, dict]:
    """Build the three pelletised cells through the model's own driver.

    The cells are keyed by the label of model.case_name, and the gas
    catalogue is rekeyed the same way so that every routine below can index it
    with the label it iterates over.
    """
    built = build_cases(random_starts=0)
    catalogue = {case_name(case): built["gas"][case] for case in CASES}
    return built["contexts"], built["states"], built["relative_sd"], catalogue


CASE_LABELS = tuple(case_name(case) for case in CASES)


def dispersion_structure(catalogue) -> pd.DataFrame:
    """Compare the dispersion of carbon dioxide with its construction.

    Within the four-species basis carbon dioxide is one minus the other three, so
    its dispersion over repeated injections is identically the dispersion of
    their sum. Treating it instead as an independent channel replaces that
    dispersion by the quadrature sum of the three. The only repeated measurement
    of the campaign is the sequence of eight injections of one bag of the blend
    at 800 degrees Celsius, so the comparison is made there, on the injections
    themselves, and on the assigned dispersion of every cell and node for
    reference.
    """
    sequence = injection_sequence()
    sd = sequence.std(axis=0, ddof=1)
    total = sequence[:, [0, 2, 3]].sum(axis=1).std(ddof=1)
    quadrature = float(np.sqrt(sd[0] ** 2 + sd[2] ** 2 + sd[3] ** 2))
    rows = [
        {
            "treatment": "blend (pelletized)",
            "T": 800.0,
            "basis": "eight injections",
            "sd_CO": sd[0],
            "sd_CO2_observed": sd[1],
            "sd_CH4": sd[2],
            "sd_H2": sd[3],
            "sd_sum_of_three": total,
            "sd_CO2_if_independent": quadrature,
            "ratio": quadrature / max(sd[1], 1e-12),
            "corr_CO_H2": float(np.corrcoef(sequence[:, 0], sequence[:, 3])[0, 1]),
        }
    ]
    for treatment in CASE_LABELS:
        gas = catalogue[treatment]
        for k, temperature in enumerate(T_NODES):
            s_ = gas.sd[k]
            rows.append(
                {
                    "treatment": treatment,
                    "T": temperature,
                    "basis": "assigned",
                    "sd_CO": s_[0],
                    "sd_CO2_observed": np.nan,
                    "sd_CH4": s_[2],
                    "sd_H2": s_[3],
                    "sd_sum_of_three": np.nan,
                    "sd_CO2_if_independent": float(np.sqrt(s_[0] ** 2 + s_[2] ** 2 + s_[3] ** 2)),
                    "ratio": np.nan,
                    "corr_CO_H2": np.nan,
                }
            )
    return pd.DataFrame(rows)


def error_model_comparison(
    contexts, states, relative_sd, draws: int
) -> pd.DataFrame:
    """Uncertainty of the reported quantities under both gas error models."""
    rows = []
    for treatment, context in contexts.items():
        reference = collect(states[treatment], context)
        # The third case doubles the assigned dispersion, because the dispersion
        # of the repeated injections bounds the experimental repeatability from
        # below and the campaign supplies no replicate pyrolysis runs.
        for model, scale in (("by_difference", 1.0), ("independent", 1.0),
                             ("by_difference_x2", 2.0)):
            rng = np.random.default_rng(RANDOM_SEED)
            samples = propagate(
                context,
                states[treatment],
                scale * relative_sd,
                draws,
                rng,
                gas_error_model=model.replace("_x2", ""),
            )
            for k, temperature in enumerate(T_NODES):
                row = {"treatment": treatment, "model": model, "T": temperature}
                for key in TRACK:
                    if key in samples:
                        row[f"{key}"] = reference[key][k]
                        row[f"{key}_sd"] = float(samples[key][:, k].std(ddof=1))
                rows.append(row)
    return pd.DataFrame(rows)


def carbon_dioxide_displacement(contexts, states, catalogue) -> pd.DataFrame:
    """Systematic displacement of the by-difference carbon dioxide.

    The by-difference construction makes a systematic error of the three measured
    channels appear entirely in carbon dioxide. The displacement applied here is
    one assigned standard deviation of the sum of those channels, applied in the
    same direction at every node, which is the worst case the construction
    admits.
    """
    rows = []
    for treatment, context in contexts.items():
        gas = catalogue[treatment]
        shift = np.sqrt(
            gas.sd[:, 0] ** 2 + gas.sd[:, 2] ** 2 + gas.sd[:, 3] ** 2
        )
        for direction in (-1.0, 0.0, 1.0):
            composition = gas.mean.copy()
            composition[:, 1] = composition[:, 1] + direction * shift
            scale = (1.0 - composition[:, 1]) / composition[:, [0, 2, 3]].sum(axis=1)
            for index in (0, 2, 3):
                composition[:, index] *= scale
            trial = replace(context, composition=composition)
            state = solve(trial, start=states[treatment].vector, restarts=2)
            indices = performance_indices(state, trial)
            bound = oxygen_bound(
                composition,
                trial.thermal.theta_nodes,
                trial.y0,
                trial.molar_mass,
                weighting=trial.thermal.raw_nodes,
            )
            rows.append(
                {
                    "treatment": treatment,
                    "displacement": direction,
                    "objective": state.objective,
                    "w_liquid_500": state.w_liquid[2],
                    "w_gas_800": state.w_gas[-1],
                    "y_t_800": state.y_t[-1],
                    "x_t_800": state.x_t[-1],
                    "CGE_800": indices["CGE"][-1],
                    "LHV_500": indices["LHV_gas"][2],
                    "w_gas_max_700": bound["w_gas_max"].to_numpy()[4],
                    "phi_tar_min_700": bound["phi_tar_min"].to_numpy()[4],
                    "phi_tar_min_800": bound["phi_tar_min"].to_numpy()[5],
                }
            )
    return pd.DataFrame(rows)


def renormalisation_bias(catalogue) -> pd.DataFrame:
    """Closed-form bias of the reported mole fractions and heating value.

    If the true gas carries a mole fraction psi of the omitted lump, the reported
    four-species fractions are the true ones divided by one minus psi, so every
    one of them is inflated by that factor, and the volumetric heating value
    computed from them omits the contribution of the lump.
    """
    rows = []
    calorific = np.array([LHV_VOLUMETRIC[s] for s in GAS_SPECIES])
    for treatment in CASE_LABELS:
        gas = catalogue[treatment]
        for scenario, ratio in C2_TO_METHANE.items():
            for k, temperature in enumerate(T_NODES):
                phi = gas.mean[k]
                lump = ratio[k] * phi[2]          # per mole of four-species gas
                psi = lump / (1.0 + lump)         # mole fraction of the true gas
                reported = phi @ calorific
                true_value = (phi @ calorific + lump * LHV_C2) / (1.0 + lump)
                rows.append(
                    {
                        "treatment": treatment,
                        "scenario": scenario,
                        "T": temperature,
                        "psi_C2": psi,
                        "inflation": 1.0 / (1.0 - psi),
                        "missing_carbon_fraction": C2_CARBON * lump
                        / (phi[0] + phi[1] + phi[2] + C2_CARBON * lump),
                        "LHV_four_species": reported,
                        "LHV_with_C2": true_value,
                        "LHV_bias": (reported - true_value) / true_value,
                    }
                )
    return pd.DataFrame(rows)


def omitted_species_budget(contexts, states) -> pd.DataFrame:
    """Re-solve the model with the omitted lump carried by the balances."""
    rows = []
    for treatment, context in contexts.items():
        reference = states[treatment]
        base = performance_indices(reference, context)
        rows.append(
            {
                "treatment": treatment,
                "scenario": "four species",
                "objective": reference.objective,
                "w_tar_400": reference.w_tar[1],
                "w_tar_800": reference.w_tar[-1],
                "w_gas_800": reference.w_gas[-1],
                "w_water_800": reference.w_water[-1],
                "y_t_800": reference.y_t[-1],
                "x_t_800": reference.x_t[-1],
                "LHV_500": base["LHV_gas"][2],
                "LHV_800": base["LHV_gas"][-1],
                "CGE_800": base["CGE"][-1],
            }
        )
        for scenario, ratio in C2_TO_METHANE.items():
            trial = replace(context, c2_to_methane=ratio)
            state = solve(trial, start=reference.vector, restarts=2)
            indices = performance_indices(state, trial)
            rows.append(
                {
                    "treatment": treatment,
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
            )
    return pd.DataFrame(rows)


def oxygen_bound_with_c2(contexts, catalogue) -> pd.DataFrame:
    """Effect of the omitted lump on the bound derived from the feedstock oxygen.

    The lump carries carbon but no oxygen, so including it lowers the oxygen
    demand per atom of gaseous carbon and raises the admissible gas yield. The
    bound reported in the manuscript is therefore the conservative one, and the
    rejection of the transferred tar parameters has to survive the correction.
    """
    rows = []
    for treatment, context in contexts.items():
        gas = catalogue[treatment]
        for scenario in ("four species",) + tuple(C2_TO_METHANE):
            composition = gas.mean.copy()
            if scenario == "four species":
                extended = composition
                lump = np.zeros(len(T_NODES))
            else:
                lump = C2_TO_METHANE[scenario] * composition[:, 2]
                extended = composition / (1.0 + lump[:, None])
            frame = oxygen_bound(
                extended,
                context.thermal.theta_nodes,
                context.y0,
                context.molar_mass,
                weighting=context.thermal.raw_nodes,
            )
            # Carbon carried by the lump enters the denominator of the oxygen
            # demand, so the demand falls in the same proportion.
            carbon_four = extended[:, :3].sum(axis=1)
            carbon_total = carbon_four + C2_CARBON * lump / (1.0 + lump)
            corrected = frame["lambda_O"].to_numpy() * carbon_four / carbon_total
            for k, temperature in enumerate(T_NODES):
                rows.append(
                    {
                        "treatment": treatment,
                        "scenario": scenario,
                        "T": temperature,
                        "lambda_O": frame["lambda_O"].to_numpy()[k],
                        "lambda_O_corrected": corrected[k],
                        "w_gas_max": frame["w_gas_max"].to_numpy()[k],
                        "phi_tar_min": frame["phi_tar_min"].to_numpy()[k],
                    }
                )
    return pd.DataFrame(rows)


def anchor_sensitivity(contexts, states) -> pd.DataFrame:
    """Response of the estimate to the anchors that were removed."""
    rows = []
    cases = {
        "reported (no condensate end anchors)": {},
        "condensate end anchors restored": {"condensate_end_anchors": True},
        "anchors halved": {"anchor_scale": 2.0},
        "anchors quartered": {"anchor_scale": 4.0},
        "no curvature penalty": {"curvature_scale": 0.0},
        "curvature penalty quadrupled": {"curvature_scale": 4.0},
        "tar monotonicity relaxed": {"relax_tar_monotonicity": True},
    }
    for treatment, context in contexts.items():
        for name, overrides in cases.items():
            trial = replace(context, **overrides)
            state = solve(trial, start=states[treatment].vector, restarts=2)
            indices = performance_indices(state, trial)
            rows.append(
                {
                    "treatment": treatment,
                    "case": name,
                    "objective": state.objective,
                    "w_liquid_500": state.w_liquid[2],
                    "w_char_800": state.w_char[-1],
                    "x_t_800": state.x_t[-1],
                    "y_t_800": state.y_t[-1],
                    "y_t_400": state.y_t[1],
                    "CGE_800": indices["CGE"][-1],
                }
            )
    return pd.DataFrame(rows)


def heating_rate_sensitivity(contexts, states) -> pd.DataFrame:
    """Re-solve the two single-residue cells on the 5 K/min thermograms of a former campaign.

    The thermograms of the estimate are recorded at the 30 K/min ramp of the
    reactor; re-solving on thermograms recorded at 5 K/min measures the effect of
    transferring the solid-yield trajectory across heating rates. The former
    campaign covers rice husk and corncob only, so the blend is not included.
    """
    from .model import THERMOGRAM_FILE_5K, build_thermal_basis, load_feedstocks

    feedstocks = load_feedstocks()
    rows = []
    for case in CASES:
        name, _ = case
        if name not in THERMOGRAM_FILE_5K:
            continue
        label = case_name(case)
        context = contexts[label]
        for ramp, basis in (
            ("30 K/min", context.thermal),
            ("5 K/min", build_thermal_basis(
                feedstocks[name], read_thermogram(DATA / THERMOGRAM_FILE_5K[name]))),
        ):
            trial = replace(context, thermal=basis)
            state = solve(trial, start=states[label].vector, restarts=2)
            indices = performance_indices(state, trial)
            rows.append({
                "treatment": label,
                "ramp": ramp,
                "theta_300": basis.theta_nodes[0],
                "theta_800": basis.theta_nodes[-1],
                "T_M": basis.peak_temperature,
                "T_on": basis.onset_temperature,
                "beta": basis.beta,
                "objective": state.objective,
                "w_liquid_500": state.w_liquid[2],
                "w_tar_400": state.w_tar[1],
                "w_gas_800": state.w_gas[-1],
                "y_t_800": state.y_t[-1],
                "x_t_800": state.x_t[-1],
                "CGE_800": indices["CGE"][-1],
            })
    return pd.DataFrame(rows)


def organic_share_of_husk() -> float:
    """Share of the organic matter of the blend contributed by rice husk."""
    from .model import BLEND_SHARE, load_feedstocks

    feedstocks = load_feedstocks()
    husk, cob = feedstocks["rice husk"], feedstocks["corncob"]
    organic_husk = BLEND_SHARE * (100.0 - husk.moisture - husk.ash)
    organic_cob = (1.0 - BLEND_SHARE) * (100.0 - cob.moisture - cob.ash)
    return organic_husk / (organic_husk + organic_cob)


def blend_additivity() -> pd.DataFrame:
    """Test the additivity of the two residues against the blend thermogram.

    The solid-yield trajectory of the blend is predicted from those of the two
    components, weighted by the organic mass each contributes, and compared with
    the trajectory measured on the blend itself.
    """
    from .model import (
        FEEDSTOCKS, build_thermal_basis, load_feedstocks, mean_thermogram,
    )

    feedstocks = load_feedstocks()
    basis, spread = {}, {}
    for name in FEEDSTOCKS:
        curve, runs = mean_thermogram(name)
        basis[name] = build_thermal_basis(feedstocks[name], curve, runs)
        spread[name] = basis[name].theta_sd

    share = organic_share_of_husk()
    predicted = share * basis["rice husk"].theta_nodes + (1.0 - share) * basis["corncob"].theta_nodes
    prediction_sd = np.sqrt(
        (share * spread["rice husk"]) ** 2 + ((1.0 - share) * spread["corncob"]) ** 2
    )
    measured = basis["blend"].theta_nodes
    measured_sd = spread["blend"]
    combined = np.sqrt(prediction_sd**2 + measured_sd**2)
    return pd.DataFrame({
        "T": T_NODES,
        "organic_share_rice_husk": share,
        "predicted": predicted,
        "predicted_sd": prediction_sd,
        "measured": measured,
        "measured_sd": measured_sd,
        "deviation": measured - predicted,
        "combined_sd": combined,
        "z": (measured - predicted) / combined,
    })


def gas_additivity(contexts, states, catalogue) -> pd.DataFrame:
    """Predict the released gas of the blend from its two components.

    Two predictions are compared with the chromatogram of the blend, which
    enters neither. The first mixes the measured mole fractions of the two
    pelletised residues in proportion to the organic mass each contributes. The
    second mixes the moles of each species released per unit of organic matter,
    which the framework estimates for each residue, and closes the result; it is
    the physically consistent one, because a residue that releases more gas
    contributes more of it to the mixture. The deviation is expressed in units
    of the dispersion assigned to the blend.
    """
    share = organic_share_of_husk()
    husk = "rice husk (pelletized)"
    blend, cob = "blend (pelletized)", "corncob (pelletized)"
    measured = catalogue[blend].mean
    sd = catalogue[blend].sd

    naive = share * catalogue[husk].mean + (1.0 - share) * catalogue[cob].mean

    def released_per_mass(label):
        state, context = states[label], contexts[label]
        moles = np.column_stack([state.n_CO, state.n_CO2, state.n_CH4, state.n_H2])
        released = np.diff(np.vstack([np.zeros(4), moles]), axis=0)
        return np.clip(released, 0.0, None) / context.molar_mass

    weighted = share * released_per_mass(husk) + (1.0 - share) * released_per_mass(cob)
    weighted = weighted / weighted.sum(axis=1, keepdims=True)

    rows = []
    for k, temperature in enumerate(T_NODES):
        for index, species in enumerate(GAS_SPECIES):
            rows.append({
                "T": temperature,
                "species": species,
                "measured": measured[k, index],
                "sd": sd[k, index],
                "husk": catalogue[husk].mean[k, index],
                "cob": catalogue[cob].mean[k, index],
                "predicted_fraction_mixing": naive[k, index],
                "predicted_yield_mixing": weighted[k, index],
                "z_fraction_mixing": (measured[k, index] - naive[k, index]) / sd[k, index],
                "z_yield_mixing": (measured[k, index] - weighted[k, index]) / sd[k, index],
            })
    return pd.DataFrame(rows)


def feedstock_contrast(contexts, states) -> pd.DataFrame:
    """Side-by-side reading of the three pelletised feedstocks."""
    rows = []
    for case in CASES:
        label = case_name(case)
        state, context = states[label], contexts[label]
        indices = performance_indices(state, context)
        rows.append({
            "feedstock": case[0],
            "objective": state.objective,
            "w_char_800": state.w_char[-1],
            "w_tar_400": state.w_tar[1],
            "w_liquid_500": state.w_liquid[2],
            "w_gas_800": state.w_gas[-1],
            "water_in_liquid_800": state.w_water[-1] / max(state.w_liquid[-1], 1e-12),
            "y_t_800": state.y_t[-1],
            "x_t_800": state.x_t[-1],
            "LHV_500": indices["LHV_gas"][2],
            "CGE_800": indices["CGE"][-1],
            "Y_H2_800": indices["Y_H2"][-1],
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    contexts, states, relative_sd, catalogue = build()

    blend_additivity().to_csv(RESULTS / "blend_additivity.csv", index=False)
    print("blend additivity written", flush=True)

    gas_additivity(contexts, states, catalogue).to_csv(
        RESULTS / "gas_additivity.csv", index=False
    )
    print("gas additivity written", flush=True)

    feedstock_contrast(contexts, states).to_csv(
        RESULTS / "feedstock_contrast.csv", index=False
    )
    print("feedstock contrast written", flush=True)

    heating_rate_sensitivity(contexts, states).to_csv(
        RESULTS / "heating_rate.csv", index=False
    )
    print("heating rate sensitivity written", flush=True)

    dispersion_structure(catalogue).to_csv(
        RESULTS / "gas_dispersion_structure.csv", index=False
    )
    print("dispersion structure written", flush=True)

    renormalisation_bias(catalogue).to_csv(
        RESULTS / "c2_renormalisation_bias.csv", index=False
    )
    print("renormalisation bias written", flush=True)

    oxygen_bound_with_c2(contexts, catalogue).to_csv(
        RESULTS / "oxygen_bound_c2.csv", index=False
    )
    print("oxygen bound with C2 written", flush=True)

    anchor_sensitivity(contexts, states).to_csv(
        RESULTS / "anchor_sensitivity.csv", index=False
    )
    print("anchor sensitivity written", flush=True)

    omitted_species_budget(contexts, states).to_csv(
        RESULTS / "c2_budget.csv", index=False
    )
    print("C2 budget written", flush=True)

    carbon_dioxide_displacement(contexts, states, catalogue).to_csv(
        RESULTS / "co2_displacement.csv", index=False
    )
    print("CO2 displacement written", flush=True)

    error_model_comparison(contexts, states, relative_sd, N_MONTE_CARLO).to_csv(
        RESULTS / "gas_error_models.csv", index=False
    )
    print("error model comparison written", flush=True)


if __name__ == "__main__":
    main()
