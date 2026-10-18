"""ccpyro: conservation-constrained estimation of pyrolysis products and tar composition.

Modules
-------
model            stoichiometric representation, closed-form elimination of the C/H/O balances, closures,
                 admissibility constraints, objective and SQP multistart solver (Algorithm 1)
identifiability  levels of evidence, profile-likelihood intervals (Algorithm 2), multistart ensembles
creck            CRECK mechanism of Debiagi et al. (2018), characterisation by reference mixtures, lumping
correlations     benchmark with a constant tar formula and penalised balances (balance-and-correlation models)
energy           energy balance of a small pyrolysis unit from the estimated products
global_potential electricity from rice husk and corncob for rural populations without access, by country
"""

__version__ = "1.0.0"
