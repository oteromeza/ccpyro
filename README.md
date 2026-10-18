# ccpyro

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23460932.svg)](https://doi.org/10.5281/zenodo.23460932)

Conservation-constrained estimation of the products of biomass pyrolysis and of the elemental composition
(CH<sub>x</sub>O<sub>y</sub>) of the tar from routine measurements: the thermogram, the proximate and ultimate
analyses and the composition of the permanent gas.

The package implements the framework of the article *Estimating tar composition from routine pyrolysis
measurements: a conservation-constrained framework with identifiability analysis* (Computers & Chemical
Engineering, submitted). It includes:

- closed-form elimination of the carbon, hydrogen and oxygen balances, which guarantees exact conservation at
  each iteration;
- the closures, the admissibility constraints that couple the temperature nodes and the oxygen bound on the
  permanent gas;
- the nonlinear program solved by sequential quadratic programming with multistart (Algorithm 1);
- profile-likelihood intervals at three levels of evidence (Algorithm 2) and Monte Carlo propagation;
- the CRECK mechanism of Debiagi et al. (2018) with the characterisation of the feedstock by reference
  mixtures, used as a benchmark;
- the benchmark with a constant tar formula and penalised balances (balance-and-correlation models);
- the energy balance of a small pyrolysis unit and the electricity that rice husk and corncob could supply to
  rural populations without access, by country.

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/oteromeza/ccpyro.git
cd ccpyro
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[notebooks,test]"
pytest -q
```

## Quick start

```python
import numpy as np
from ccpyro import model as mp

feed = mp.load_feedstocks()
catalogue, _ = mp.load_gas_data()
case = ("rice husk", "pelletized")
curve, runs = mp.mean_thermogram(case[0])
thermal = mp.build_thermal_basis(feed[case[0]], curve, runs)
gas = catalogue[case]
ctx = mp.ModelContext(feedstock=feed[case[0]], thermal=thermal, composition=gas.mean,
                      composition_sd=gas.sd, alpha=mp.ALPHA_TRANSFERRED)
state = mp.solve(ctx)                       # SQP from the closure-based start
print(state.w_tar, state.x_t, state.y_t)    # tar yield and CHxOy by temperature node
```

## Reproducing the article

`run_all.py` runs the complete chain and then executes the notebooks. The results, the figure data, the logs and
the figures are not part of the repository; every run writes them again to `results/`, `figdata/`, `logs/` and
`figures/`.

```bash
python run_all.py --quick      # reduced settings: checks the complete chain (about 1.5 h with two processors)
python run_all.py              # settings of the article
```

| Stage | Jobs | Content |
|---|---|---|
| `estimate` | 1 | Two sequential passes and 40 admissible random starts per feedstock (Algorithm 1) |
| `ensembles` | 3 | 160 random starts of the complete estimator per feedstock |
| `adopt` | 3 | The best state of each ensemble replaces the estimate when it lowers the objective |
| `model` | 18 | Monte Carlo (300 draws), sensitivity analyses, profile likelihood (Algorithm 2), ensembles of the measurements and of the closures, spectra, supporting analyses and balance-and-correlation benchmark |
| `merge` | 2 | Joins the files of each feedstock and extends the intervals with the ensemble members |
| `post` | 3 | CRECK benchmark, energy balance of the unit, global estimate and figure data |
| `notebooks` | 9 | Executes the notebooks in place |

Independent jobs run in parallel (`--jobs`, by default the number of processors minus one), each with a single
thread. A failed job does not stop the other jobs of its stage, and the run stops at the end of that stage.
`--from <stage>` resumes the last run: the jobs that already finished are not repeated, and every stage after the
first one that executes a job is repeated in full. `--force` repeats the jobs even if they finished. The log of each
job is in `logs/`, and `results/run_summary.json` records the settings, the platform and the duration of every job.

## Notebooks

| Notebook | Content |
|---|---|
| `01_worked_example_single_node.ipynb` | One temperature node by hand: parent formula, elimination, closures, constraints and residuals (Supplementary Section S1) |
| `02_estimation_and_figures.ipynb` | Estimation of the three feedstocks from the reported estimate, conservation check and figures |
| `03_identifiability.ipynb` | Profile-likelihood intervals at the three levels of evidence, and one profile recomputed |
| `04_monte_carlo.ipynb` | Uncertainty propagation |
| `05_creck_benchmark.ipynb` | Characterisation by reference mixtures and integration of the CRECK mechanism |
| `06_balance_and_correlation.ipynb` | Constant-formula benchmark with penalised balances |
| `07_energy.ipynb` | Energy balance of the unit |
| `08_global_potential.ipynb` | Electricity for rural populations without access, by country |
| `09_figures.ipynb` | pgfplots sources and PDFs of the data figures of the article (`figures/`), compiled with the common style `figstyle.tex` that the notebook writes |

The notebooks read the results that `run_all.py` writes to `results/` and write their own outputs to
`notebooks/output`; `09_figures.ipynb` writes the data figures of the article to `figures/`. The notebooks
`01_worked_example_single_node.ipynb` and `02_estimation_and_figures.ipynb` also run on a fresh installation,
from the reported estimate that ships with the package.

## Data

The experimental data ship with the package in `src/ccpyro/data` and are also archived in Zenodo,
https://doi.org/10.5281/zenodo.22922087:

- `pellets_composition.xlsx`: laboratory report of the pellets (proximate analysis as received, ultimate analysis
  on the dry basis, heating values, biochemical composition and physical properties, with all replicates);
- `TGA_30Cmin.csv`: thermograms at 30 °C/min (three runs per feedstock); `TGA_rice_husk.csv` and
  `TGA_corncob.csv`: thermograms of an earlier campaign at 5 °C/min, used only in a sensitivity analysis;
- `gas_composition.xlsx`: composition of the gas released over each stage;
- `literature_char.csv` and `literature_tar.csv`: published elemental compositions of chars and water-free tars from the slow
  pyrolysis of lignocellulosic residues, used in the Van Krevelen diagram and the CRECK benchmark of the article;
- `global/`: production of paddy rice and maize (FAOSTAT, mean of 2022-2024) and rural access to electricity
  and rural population (World Bank, World Development Indicators), refreshed with `tools/fetch_global_data.py`.

`src/ccpyro/data/best_vectors.npz` holds the best decision vector found for each feedstock, which is the reported estimate and
the starting point that `run_all.py` refines. Outputs are written to `./results` and `./figdata` of the working
directory unless the environment variables `CCPYRO_RESULTS` and `CCPYRO_FIGDATA` say otherwise; `CCPYRO_DATA`
points the package to another data folder.

## How to cite

See `CITATION.cff`. Please cite the article and the archived version of the software used; version 1.0.0 is
https://doi.org/10.5281/zenodo.23460932.

## License

Code: MIT. Data: CC BY 4.0.
