"""Characterisation read from the laboratory report: basis conversion and consistency with the heating value."""
import numpy as np
import pandas as pd

from ccpyro import model as mp


def test_dry_basis_conversion():
    report = mp.DATA / mp.LAB_REPORT
    proximate = pd.read_excel(report, sheet_name="proximate").dropna(subset=["treatment"])
    ultimate = pd.read_excel(report, sheet_name="ultimate").dropna(subset=["treatment"])
    for name, feed in mp.load_feedstocks().items():
        code = mp.GAS_LABEL[name]
        p = proximate[proximate.treatment == code]
        u = ultimate[ultimate.treatment == code]
        ash_dry = p.AC.mean() / (1 - p.MC.mean() / 100)
        daf = feed.daf_composition
        assert np.isclose(daf["C"], u.C.mean() / (1 - ash_dry / 100))
        assert np.isclose(daf["H"], u.H.mean() / (1 - ash_dry / 100))
        assert np.isclose(daf["C"] + daf["H"] + daf["O"] + daf["N"], 100.0)


def test_ultimate_analysis_reproduces_measured_heating_value():
    """Channiwala and Parikh (2002) on the dry basis, with the ash term, against the bomb calorimeter."""
    measured = mp.load_heating_values()
    for name, feed in mp.load_feedstocks().items():
        code = mp.GAS_LABEL[name]
        a = 100 * feed.ash_dry
        daf = feed.daf_composition
        c, h, n = (daf[e] * (1 - a / 100) for e in ("C", "H", "N"))
        o = 100 - c - h - n - a
        hhv = 0.3491 * c + 1.1783 * h - 0.1034 * o - 0.0151 * n - 0.0211 * a
        assert abs(hhv / measured.loc[code, "HHV_mean"] - 1) < 0.01
