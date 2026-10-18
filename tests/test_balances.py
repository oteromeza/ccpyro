"""Exact conservation in the framework and in the CRECK mechanism; characterisation against Ranzi (2017)."""
import numpy as np

from ccpyro import creck
from ccpyro import model as mp


def test_reported_estimate_conserves_elements():
    feed = mp.load_feedstocks()
    catalogue, _ = mp.load_gas_data()
    for case in mp.CASES:
        curve, runs = mp.mean_thermogram(case[0])
        thermal = mp.build_thermal_basis(feed[case[0]], curve, runs)
        gas = catalogue[case]
        ctx = mp.ModelContext(feedstock=feed[case[0]], thermal=thermal, composition=gas.mean,
                              composition_sd=gas.sd, alpha=mp.ALPHA_TRANSFERRED)
        s = mp.unpack(np.load(mp.BEST_VECTORS)[mp.case_name(case)], ctx)
        carbon = s.n_c + s.n_t + s.n_CO + s.n_CO2 + s.n_CH4
        hydrogen = s.x_c * s.n_c + s.x_t * s.n_t + 2 * s.n_H2O + 4 * s.n_CH4 + 2 * s.n_H2
        oxygen = s.y_c * s.n_c + s.y_t * s.n_t + s.n_H2O + s.n_CO + 2 * s.n_CO2
        assert np.allclose(carbon, 1.0, atol=1e-12)
        assert np.allclose(hydrogen, ctx.x0, atol=1e-12)
        assert np.allclose(oxygen, ctx.y0, atol=1e-12)
        assert np.allclose(s.w_char + s.w_tar + s.w_water + s.w_gas, 1.0, atol=1e-12)


def test_creck_reactions_balance():
    mech = creck.load_mechanism()
    assert len(mech.reactions) == 32
    for r in mech.reactions:
        assert np.all(np.abs(r.imbalance()) < 1e-9), r.text


def test_characterisation_reproduces_ranzi_2017_poplar():
    # elemental composition implied by the published species of the hybrid poplar example
    atoms, mw = creck._species_formula()
    species = np.array([.3627, .1970, .0489, .0017, .3181, .0716, 0.0])
    elements = (species / mw) @ atoms * np.array([12.011, 1.008, 15.999])
    result = creck.characterise(*(elements / elements.sum()), epsilon=0.8)
    assert np.allclose(result["rm"], [0.5597, 0.0020, 0.4384], atol=2e-4)
    got = np.array([result["species"][k] for k in creck.CHAR_SPECIES])
    assert np.allclose(got, species, atol=2e-4)
