"""Reproduce every result of the article and execute the notebooks.

usage (from the root of the repository):
    python run_all.py                  complete run with the settings of the article
    python run_all.py --jobs 6         limit the number of simultaneous processes
    python run_all.py --quick          reduced settings, to check the complete chain in a fraction of the full time
    python run_all.py --from model     resume from a stage (estimate, ensembles, adopt, model, merge, post, notebooks);
                                       the jobs already finished in the last run are not repeated
    python run_all.py --only post      run a single stage

Results are written to ./results and ./figdata, the log of every job to ./logs, and the executed notebooks replace
those in ./notebooks. The run is summarised in results/run_summary.json. Independent jobs run in parallel, each
in its own process with a single thread.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS, FIGDATA, LOGS, TOOLS, NOTEBOOKS = (ROOT / d for d in ("results", "figdata", "logs", "tools", "notebooks"))
CELLS = ("huskp", "blendp", "cobp")
STAGES = ("estimate", "ensembles", "adopt", "model", "merge", "post", "notebooks")

FULL = {"random_starts": 40, "ensemble_starts": 160, "mc_draws": 300, "profile_points": 13, "benchmark_starts": 8}
QUICK = {"random_starts": 2, "ensemble_starts": 6, "mc_draws": 8, "profile_points": 4, "benchmark_starts": 1}


def environment(settings: dict) -> dict:
    env = os.environ.copy()
    env.update({
        "CCPYRO_RESULTS": str(RESULTS),
        "CCPYRO_FIGDATA": str(FIGDATA),
        "CCPYRO_MC_DRAWS": str(settings["mc_draws"]),
        "CCPYRO_PROFILE_POINTS": str(settings["profile_points"]),
        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    return env


def jobs_of(stage: str, s: dict) -> list[tuple[str, list[str]]]:
    py = sys.executable

    def tool(name, *args):
        return [py, str(TOOLS / name), *map(str, args)]

    def ident(*args):
        return [py, "-m", "ccpyro.identifiability", *map(str, args)]

    if stage == "estimate":
        return [("estimate", tool("estimate.py", "--random-starts", s["random_starts"]))]
    if stage == "ensembles":
        return [(f"multistart_estimator_{c}", ident("--cell", c, "--stage", "multistart", "--criterion",
                                                     "estimator", "--starts", s["ensemble_starts"])) for c in CELLS]
    if stage == "adopt":
        return [(f"adopt_{c}", tool("adopt_ensemble_best.py", "--cell", c)) for c in CELLS]
    if stage == "model":
        # longest jobs first, so that the pool finishes as early as possible
        out = [("model", [py, "-m", "ccpyro.model"]), ("analyses", tool("analyses_chain.py"))]
        out += [(f"profiles_{c}_{v}", tool("run_profiles.py", c, v)) for c in CELLS for v in ("y_t", "x_t")]
        out += [(f"multistart_{k}_{c}", ident("--cell", c, "--stage", "multistart", "--criterion", k, "--starts",
                                                s["ensemble_starts"])) for k in ("closures", "measurements")
                for c in CELLS]
        out += [(f"spectra_{c}", ident("--cell", c, "--stage", "spectra")) for c in CELLS]
        out += [("balance_correlation", [py, "-c", "import sys; from ccpyro import correlations as c; "
                                         f"c.run(sys.argv[1], starts={s['benchmark_starts']})", str(RESULTS)])]
        return out
    if stage == "merge":
        return [("merge", tool("merge_identifiability.py")), ("combine", tool("combine_intervals.py"))]
    if stage == "post":
        return [("creck", tool("creck_outputs.py")), ("energy_global", tool("global_outputs.py")),
                ("figdata", tool("build_figdata.py"))]
    if stage == "notebooks":
        return [(f"notebook_{nb.stem}", [py, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                                         "--ExecutePreprocessor.timeout=7200", str(nb)])
                for nb in sorted(NOTEBOOKS.glob("*.ipynb"))]
    raise ValueError(stage)


SEQUENTIAL = {"adopt", "merge"}   # these jobs write to the same files, one after the other
ACTIVE: list = []


def marker(name: str) -> Path:
    return LOGS / f"{name}.done"


def run_stage(stage: str, settings: dict, env: dict, max_jobs: int, record: dict, run_id: str,
              allow_skip: bool) -> tuple[bool, bool]:
    """Run the jobs of one stage in parallel and return (success, whether any job was executed).

    A failed job does not stop the others of the stage; the run stops after the stage. A job that already finished
    in the same run (same run identifier) is not repeated, as long as no earlier stage was executed again.
    """
    jobs = jobs_of(stage, settings)
    limit = 1 if stage in SEQUENTIAL else max_jobs
    pending, running, failed = [], {}, []
    for name, argv in jobs:
        done = marker(name)
        if allow_skip and done.exists() and done.read_text(encoding="utf-8").strip() == run_id:
            print(f"  skipped  {name} (already finished in this run)", flush=True)
        else:
            pending.append((name, argv))
    start = time.time()
    ran_any = bool(pending)
    print(f"\n=== {stage}: {len(pending)} job(s) to run, up to {limit} at a time", flush=True)
    while pending or running:
        while pending and len(running) < limit:
            name, argv = pending.pop(0)
            log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
            cwd = NOTEBOOKS if stage == "notebooks" else ROOT
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
            ACTIVE.append(proc)
            running[name] = (proc, log, time.time())
            print(f"  started  {name}", flush=True)
        for name in list(running):
            proc, log, t0 = running[name]
            if proc.poll() is None:
                continue
            log.close()
            minutes = (time.time() - t0) / 60
            record["jobs"][name] = {"stage": stage, "minutes": round(minutes, 2), "returncode": proc.returncode,
                                    "run_id": run_id, "executed": datetime.now().isoformat(timespec="seconds")}
            if proc.returncode == 0:
                marker(name).write_text(run_id, encoding="utf-8")
                print(f"  finished {name} in {minutes:.1f} min", flush=True)
            else:
                failed.append(name)
                print(f"  FAILED   {name} ({proc.returncode}) after {minutes:.1f} min; see logs/{name}.log", flush=True)
            del running[name]
        time.sleep(2)
    record["stages"][stage] = round((time.time() - start) / 60, 2)
    if failed:
        print(f"\nStage {stage} failed: {', '.join(failed)}. See logs/<job>.log. After the fix, "
              f"`python run_all.py --from {stage}{' --quick' if record.get('quick') else ''}` repeats only the "
              f"failed jobs and continues.")
        return False, ran_any
    print(f"=== {stage} done in {(time.time() - start) / 60:.1f} min", flush=True)
    return True, ran_any


def check_dependencies(with_notebooks: bool) -> None:
    missing = []
    modules = ["numpy", "scipy", "pandas", "openpyxl"]
    if with_notebooks:
        modules += ["nbformat", "nbconvert", "ipykernel", "matplotlib"]
    for module in modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        sys.exit("Missing packages: " + ", ".join(missing) + '.\nInstall them with: pip install -e ".[notebooks,test]"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--from", dest="start", choices=STAGES, default=STAGES[0])
    parser.add_argument("--only", choices=STAGES)
    parser.add_argument("--no-notebooks", action="store_true")
    parser.add_argument("--force", action="store_true", help="repeat the jobs even if they finished in the last run")
    arguments = parser.parse_args()

    stages = [arguments.only] if arguments.only else list(STAGES[STAGES.index(arguments.start):])
    if arguments.no_notebooks and "notebooks" in stages:
        stages.remove("notebooks")
    check_dependencies("notebooks" in stages)
    for folder in (RESULTS, FIGDATA, LOGS):
        folder.mkdir(exist_ok=True)
    settings = QUICK if arguments.quick else FULL
    env = environment(settings)
    summary_path = RESULTS / "run_summary.json"
    record = {"jobs": {}, "stages": {}}
    # A complete run gets a new identifier; --from and --only continue the last run of the same kind, so that
    # the jobs it already finished are not repeated.
    id_file = LOGS / ("run_id_quick.txt" if arguments.quick else "run_id_full.txt")
    resuming = bool(arguments.only) or arguments.start != STAGES[0]
    if resuming and id_file.exists() and not arguments.force:
        run_id = id_file.read_text(encoding="utf-8").strip()
    else:
        run_id = f"{'quick' if arguments.quick else 'full'}-{datetime.now():%Y%m%d-%H%M%S}"
        id_file.write_text(run_id, encoding="utf-8")
    if summary_path.exists() and resuming:
        previous = json.loads(summary_path.read_text())
        record["jobs"].update(previous.get("jobs", {}))
        record["stages"].update(previous.get("stages", {}))
    record.update({"run_id": run_id, "started": datetime.now().isoformat(timespec="seconds"), "settings": settings,
                   "quick": arguments.quick, "python": sys.version.split()[0], "platform": platform.platform(),
                   "processors": os.cpu_count(), "max_jobs": arguments.jobs})
    print(f"ccpyro: {'QUICK CHECK' if arguments.quick else 'full run'} with up to {arguments.jobs} processes; "
          f"stages: {', '.join(stages)}", flush=True)
    ok, executed = True, False
    try:
        for stage in stages:
            # once a stage has executed any job, the later stages are repeated in full, because their inputs changed
            ok, ran_any = run_stage(stage, settings, env, arguments.jobs, record, run_id, allow_skip=not executed)
            executed = executed or ran_any
            record["finished"] = datetime.now().isoformat(timespec="seconds")
            summary_path.write_text(json.dumps(record, indent=2))
            if not ok:
                break
    except KeyboardInterrupt:
        for proc in ACTIVE:
            if proc.poll() is None:
                proc.terminate()
        sys.exit("\nInterrupted; the running jobs were stopped. Resume with --from <stage>.")
    if ok:
        print(f"\nAll stages completed. Summary in {summary_path.relative_to(ROOT)}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
