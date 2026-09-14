#!/usr/bin/env python
"""Run the GCV, blind-test and variant-database suites to completion.

Every underlying script already resumes at the seed level. This adds the layer
above: the full job matrix, completion detection, and two bounded worker pools
so a reproduction run uses the whole machine instead of one core at a time.

    conda run -n ppi python src/run_benchmarks.py --status
    conda run -n ppi python src/run_benchmarks.py --suite gcv --jobs 12 --gpus 0,1,2,3
    conda run -n ppi python src/run_benchmarks.py --suite all --dry-run

Two pools, because the methods split cleanly by the resource they saturate:

    cpu   MINT, PPLM, SWING, SAAMBE-3D      sklearn / XGBoost / subprocess
    gpu   MutPred-PPI, eSIG-Net, MutPPI     torch, one job per GPU

`--threads` caps BLAS/OpenMP threads per job and defaults to 1. This is not
just a tidiness knob. The libraries default to one thread per core, so on a
many-core machine a single small MLP fit spends most of its time in OpenMP
barriers: measured on a 72-core host, one fit took 394 s at 72 threads and
113 s at 1 -- 3.5x slower for using 72x the cores. Capping threads and running
many jobs side by side is strictly better than the reverse.

Thread count is not numerically free: OpenBLAS partitions reductions by team
size, and the MLP's Adam trajectory amplifies that over 100 iterations, so GCV
AUCs move in the 4th decimal between thread settings. Pick one value for a whole
suite rather than mixing, and re-run rather than resuming across a change.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from paths import GCV_RESULTS_DIR, REPO_ROOT, VARCHAMP_BLIND_TEST_DIR
from utils.gcv_common import DATASET_CONFIGS, dataset_name

# The three datasets every comparison figure reads (Fig 3, S1, S7).
GCV_DATASETS = [dataset_name(b) for b in
                ("sahni_only", "sahni_fragoza", "sahni_fragoza_varchamp_all")]
VARIANT_DBS = ["gnomad", "clinvar", "cosmic", "hgmd", "neurodev", "asd"]

# Fig S4 plots MutPred-PPI against its own ablations on sahni_fragoza. The
# headline `megascale_all` run is part of the main GCV matrix; these are the
# other arms, and without them that figure cannot be drawn.
#
# `Prior Best` in ABLATION_DISPLAY_NAMES is NOT here: it is the FoldX-pretrained
# RECOMB 2024 model, whose result stem carries no ablation suffix and which the
# current code cannot reproduce. It has to come from the archived v1.0 results.
ABLATIONS = [
    "freeze_mut_processor",      # Freeze Mutation Processor
    "freeze_gat",                # Freeze GAT
    "megascale_head",            # Freeze Both (head-only probe)
    "megascale_all_no-gat",      # No GAT
    "megascale_all_no-mut",      # No Mutation Processor
    "megascale_all_wt-emb",      # WT Embedding
    "scratch",                   # No Pretrain
    "prior_best",                # Prior Best (prior published model, archived)
    "pretrain_zero_shot",        # stability pretrain, untrained on PPI
]
ABLATION_DATASET = dataset_name("sahni_fragoza")
N_GCV = 30
_PY = "python"


@dataclass
class Job:
    name: str
    kind: str                      # "cpu" | "gpu"
    argv: list[str]
    is_done: Callable[[], bool]
    gpu_flag: str = "--device"     # how this job takes its GPU
    env: dict[str, str] = field(default_factory=dict)


# ── completion predicates ─────────────────────────────────────────────────────

def _seeds_done(stem: str, n: int = N_GCV) -> Callable[[], bool]:
    """A run_gcv job is done when its micro_aucs array holds every seed."""
    def check() -> bool:
        p = Path(GCV_RESULTS_DIR) / f"{stem}_micro_aucs.npy"
        if not p.exists():
            return False
        try:
            return np.load(p).shape[0] >= n
        except Exception:
            return False
    return check


def _file_exists(path: Path) -> Callable[[], bool]:
    return lambda: path.exists()


def _glob_exists(directory: Path, pattern: str) -> Callable[[], bool]:
    return lambda: any(directory.glob(pattern))


# ── job matrices ──────────────────────────────────────────────────────────────

def gcv_jobs(datasets: list[str], ablation: str = "megascale_all") -> list[Job]:
    jobs: list[Job] = []
    for ds in datasets:
        E = "src/evaluation"
        jobs.append(Job(
            f"mutpredppi_{ds}", "gpu",
            [_PY, f"{E}/mutpred_ppi_gcv.py", "--dataset", ds,
             "--ablation", ablation, "--n-gcv", str(N_GCV)],
            _seeds_done(f"MutPredPPI_{ds}_{ablation}")))
        for pred in ("seq_diff", "site_diff"):
            jobs.append(Job(
                f"mint_{pred}_{ds}", "cpu",
                [_PY, f"{E}/mint_cv.py", "--dataset", ds,
                 "--predictor", pred, "--n-gcv", str(N_GCV)],
                _seeds_done(f"MINT_{pred}_{ds}")))
            jobs.append(Job(
                f"pplm_{pred}_{ds}", "cpu",
                [_PY, f"{E}/pplm_cv.py", "--dataset", ds,
                 "--predictor", pred, "--n-gcv", str(N_GCV)],
                _seeds_done(f"PPLM_{pred}_{ds}")))
        for tp, code in ((False, "_no_test_pretrain"), (True, "_test_pretrain")):
            argv = [_PY, f"{E}/swing_gcv.py", "--dataset", ds]
            if tp:
                argv.append("--test-pretrain")
            jobs.append(Job(f"swing{code}_{ds}", "cpu", argv,
                            _seeds_done(f"SWING_{ds}{code}")))
        jobs.append(Job(
            f"esignet_{ds}", "gpu",
            [_PY, f"{E}/esignet_cv.py", "--dataset", ds, "--n-gcv", str(N_GCV)],
            _seeds_done(f"ESigNet_{ds}")))
        # SKEMPI-pretrained: a single scoring pass, no seeds.
        jobs.append(Job(
            f"saambe3d_{ds}", "cpu",
            [_PY, f"{E}/saambe3d_cv.py", "--dataset", ds,
             "--model-type", "regression"],
            _file_exists(Path(GCV_RESULTS_DIR) / f"{ds}_SAAMBE-3D_preds.npy")))
        for model, stem in ((0, "MutPPI"), (1, "MutPPIPlus")):
            jobs.append(Job(
                f"{stem.lower()}_{ds}", "gpu",
                [_PY, f"{E}/mutppi_cv.py", "--dataset", ds, "--model", str(model)],
                _file_exists(Path(GCV_RESULTS_DIR) / f"{ds}_{stem}_preds.npy")))
    return jobs


def blind_test_jobs() -> list[Job]:
    B = Path(VARCHAMP_BLIND_TEST_DIR)
    R = "src/evaluation/run_varchamp_blind_test.py"
    specs = [
        ("mutpredppi", [], "gpu", "MutPred-PPI*_c3_preds.npy"),
        ("esignet",    [], "gpu", "eSIG-Net*_c3_preds.npy"),
        ("swing",      [], "cpu", "SWING (Sahni*_c3_preds.npy"),
        # Upstream SWING's own configuration: one Doc2Vec fitted over train+test
        # sequences (no labels), reused everywhere -- so it is also much cheaper
        # than blind-test mode, which refits per fold. Reported separately
        # because the shared corpus is a representational leak.
        ("swing",      ["--test-pretrain"], "cpu",
         "SWING (test pretrain*_c3_preds.npy"),
        ("saambe3d",   [], "cpu", "SAAMBE-3D*_c3_preds.npy"),
        ("mutppi",     [], "gpu", "MutPPI (*_c3_preds.npy"),
        ("mutppiplus", [], "gpu", "MutPPIPlus*_c3_preds.npy"),
    ]
    jobs = [
        Job(f"blind_{m}" + ("_test_pretrain" if extra else ""), kind,
            [_PY, R, "--method", m, *extra],
            _glob_exists(B, pat))
        for m, extra, kind, pat in specs
    ]
    for method in ("mint", "pplm"):
        for pred in ("seq_diff", "site_diff"):
            jobs.append(Job(
                f"blind_{method}_{pred}", "cpu",
                [_PY, R, "--method", method, "--predictor", pred],
                _glob_exists(B, f"{method.upper()}_{pred}*_c3_preds.npy")))
    # The supplementary training-set comparison (fig:varchamp_training_set_comparison)
    # is MutPred-PPI trained on Sahni alone against the same model trained on
    # Sahni+Fragoza, both scored on the same VarChAMP test set. It needs exactly one
    # extra array -- the other half is `blind_mutpredppi` above -- so this is a single
    # job, not a second pass over every method. Only the trained methods have a
    # training set to vary at all; SAAMBE-3D/MutPPI/MutPPI+ are SKEMPI-pretrained and
    # would produce identical output.
    jobs.append(Job(
        "blind_mutpredppi_sahni_only", "gpu",
        [_PY, R, "--method", "mutpredppi",
         "--train-dataset", dataset_name("sahni_only")],
        _glob_exists(B, "MutPred-PPI (sahni,*_c3_preds.npy")))
    return jobs


def variant_db_jobs(dbs: list[str]) -> list[Job]:
    R = "src/variant_db_inference/run_variant_db_inference.py"
    # Output lives beside the repo, one directory per database.
    base = Path(os.environ.get("MUTPRED_DATA_ROOT", str(Path(REPO_ROOT).parent)))
    jobs = []
    for db in dbs:
        out = base / db / "mutpred_ppi_predictions.tsv"
        # The sentinel is written by _report() only after the row iterator is
        # exhausted. A row count cannot substitute for it: each database leaves a
        # different share of rows unscoreable, so no threshold means "done" for
        # all of them (neurodev finishes at 94.6% of its table, hgmd at 99.8%).
        jobs.append(Job(f"inference_{db}", "gpu",
                        [_PY, R, "--dataset", db],
                        _file_exists(Path(f"{out}.complete"))))
    return jobs


# ── execution ─────────────────────────────────────────────────────────────────

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(job: Job, gpu: str | None, logdir: Path, threads: int,
         dry_run: bool) -> tuple[str, bool]:
    argv = list(job.argv)
    if gpu is not None:
        argv += [job.gpu_flag, f"cuda:{gpu}"]
    if dry_run:
        _log(f"DRY-RUN {job.name}: {' '.join(argv)}")
        return job.name, True

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    env.update(job.env)

    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"{job.name}.log"
    _log(f"START {job.name}" + (f" (cuda:{gpu})" if gpu else ""))
    t0 = time.time()
    with open(logfile, "a") as fh:
        rc = subprocess.run(argv, cwd=str(REPO_ROOT), env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    dt = (time.time() - t0) / 60
    ok = rc == 0
    _log(f"{'DONE ' if ok else 'FAILED'} {job.name} ({dt:.1f} min)"
         + ("" if ok else f" rc={rc} -- see {logfile}"))
    return job.name, ok


def run_pool(jobs: list[Job], workers: int, gpus: list[str] | None,
             logdir: Path, threads: int, dry_run: bool) -> list[str]:
    """Run jobs with bounded concurrency, handing each GPU job a free device."""
    if not jobs:
        return []
    free_gpus: list[str] = list(gpus or [])
    gpu_lock = threading.Lock()

    def take_gpu() -> str | None:
        if not free_gpus:
            return None
        with gpu_lock:
            return free_gpus.pop(0) if free_gpus else None

    def give_gpu(g: str | None) -> None:
        if g is not None:
            with gpu_lock:
                free_gpus.append(g)

    def task(job: Job) -> tuple[str, bool]:
        g = take_gpu() if job.kind == "gpu" else None
        try:
            return _run(job, g, logdir, threads, dry_run)
        finally:
            give_gpu(g)

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(task, j) for j in jobs]):
            name, ok = fut.result()
            if not ok:
                failed.append(name)
    return failed


def ablation_jobs(dataset: str = ABLATION_DATASET) -> list[Job]:
    """MutPred-PPI ablation arms for Fig S4."""
    return [
        Job(f"ablation_{ab}_{dataset}", "gpu",
            [_PY, "src/evaluation/mutpred_ppi_gcv.py", "--dataset", dataset,
             "--ablation", ab, "--n-gcv", str(N_GCV)],
            _seeds_done(f"MutPredPPI_{dataset}_{ab}"))
        for ab in ABLATIONS
    ]


def build_jobs(suite: str, datasets: list[str], dbs: list[str]) -> list[Job]:
    jobs: list[Job] = []
    if suite in ("gcv", "all"):
        jobs += gcv_jobs(datasets)
    if suite in ("ablation", "all"):
        jobs += ablation_jobs()
    if suite in ("blind-test", "all"):
        jobs += blind_test_jobs()
    if suite in ("variant-db", "all"):
        jobs += variant_db_jobs(dbs)
    return jobs


def print_status(jobs: list[Job]) -> None:
    done = [j for j in jobs if j.is_done()]
    todo = [j for j in jobs if not j.is_done()]
    print(f"\n{len(done)}/{len(jobs)} jobs complete\n")
    for j in jobs:
        print(f"  [{'x' if j.is_done() else ' '}] {j.kind:<3} {j.name}")
    print(f"\n{len(todo)} remaining\n")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run the GCV / blind-test / variant-DB suites to completion.")
    p.add_argument("--suite", default="gcv",
                   choices=["gcv", "ablation", "blind-test", "variant-db", "all"])
    p.add_argument("--datasets", default=",".join(GCV_DATASETS),
                   help="Comma-separated GCV datasets.")
    p.add_argument("--dbs", default=",".join(VARIANT_DBS),
                   help="Comma-separated variant databases.")
    p.add_argument("--jobs", type=int, default=12,
                   help="Concurrent CPU-pool jobs (default: 12).")
    p.add_argument("--gpus", default="",
                   help="Comma-separated GPU ids for the GPU pool, e.g. 0,1,2,3. "
                        "Empty means run GPU jobs on the default device, one at a time.")
    p.add_argument("--threads", type=int, default=1,
                   help="BLAS/OpenMP threads per job (default: 1). Changing this "
                        "shifts GCV AUCs in the 4th decimal -- keep it fixed "
                        "across a suite.")
    p.add_argument("--logdir", default=str(Path(REPO_ROOT) / "logs" / "benchmarks"))
    p.add_argument("--only", default="",
                   help="Substring filter on job names.")
    p.add_argument("--status", action="store_true", help="Print status and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    p.add_argument("--rerun", action="store_true",
                   help="Run jobs even if their outputs already look complete.")
    args = p.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    for d in datasets:
        if d not in DATASET_CONFIGS:
            p.error(f"unknown dataset {d!r}; choose from {sorted(DATASET_CONFIGS)}")

    jobs = build_jobs(args.suite, datasets, [d for d in args.dbs.split(",") if d])
    if args.only:
        jobs = [j for j in jobs if args.only in j.name]

    if args.status:
        print_status(jobs)
        return 0

    pending = jobs if args.rerun else [j for j in jobs if not j.is_done()]
    if not pending:
        print("Nothing to do -- every job's output is already complete.")
        return 0

    gpus = [g for g in args.gpus.split(",") if g]
    cpu_jobs = [j for j in pending if j.kind == "cpu"]
    gpu_jobs = [j for j in pending if j.kind == "gpu"]
    logdir = Path(args.logdir)

    _log(f"{len(pending)} job(s) to run "
         f"({len(cpu_jobs)} cpu, {len(gpu_jobs)} gpu); "
         f"threads={args.threads}, cpu workers={args.jobs}, "
         f"gpus={gpus or 'default'}")

    # Both pools run concurrently; the GPU pool is bounded by device count.
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=2) as top:
        futs = [
            top.submit(run_pool, cpu_jobs, args.jobs, None,
                       logdir, args.threads, args.dry_run),
            top.submit(run_pool, gpu_jobs, max(len(gpus), 1), gpus,
                       logdir, args.threads, args.dry_run),
        ]
        for f in futs:
            failed += f.result()

    if failed:
        _log(f"{len(failed)} job(s) FAILED: {', '.join(failed)}")
        return 1
    _log("All jobs complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
