"""Supporting analyses of the article, in the order in which they depend on each other.

1. ccpyro.analyses: additivity of the blend, heating rate, error budget of the gas, omitted C2 species, anchors.
2. c2_budget_continuation.py: omitted-species budget recomputed with a continuation in the lump ratio.
3. c2_budget_retry.py: retry of the scenarios that the continuation did not bring into a good basin.
"""
import runpy
from pathlib import Path

from ccpyro import analyses

HERE = Path(__file__).resolve().parent
analyses.main()
for script in ("c2_budget_continuation.py", "c2_budget_retry.py"):
    print(f"--- {script}", flush=True)
    runpy.run_path(str(HERE / script), run_name="__main__")
