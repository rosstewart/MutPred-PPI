"""Shared scaffold for the MINT / PPLM seq-diff / site-diff GCV scripts.

Both methods train a small sklearn MLP on a precomputed embedding cache and
differ only in the method name, cache module, predictor classes, and cache key
schema. This module holds the common argparse skeleton, cache audit, and run()
body so mint_cv.py and pplm_cv.py each reduce to a thin wrapper.

The two cache schemas are not interchangeable, and the difference is a
space/compute tradeoff made at precompute time rather than an accident:

  MINT  stores pre-reduced arrays under flat string keys. `mean_{a}_{b}` is
        already mean-pooled over all La+Lb residues, so the reduction happens
        once at precompute. site_diff additionally needs per-residue chain A,
        held under a separate `res_*` key.
  PPLM  stores raw per-residue matrices for both chains under one key, because
        embed_B differs between WT and MUT (chain A's substituted token reaches
        chain B through cross-attention), so the partner mean must be recomputed
        per variant rather than shared across the pair.

Callers therefore supply their own key builders and validity predicate; the
counting, reporting and abort logic below is identical for both.

eSIG-Net is structurally different (GPU training, no --predictor, pre-warms
573-dim features) and keeps its own script (esignet_cv.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from paths import GCV_RESULTS_DIR, TRAINING_EVAL_DIR
from utils.gcv_common import (
    dataset_arg,
    DATASET_CONFIGS,
    DatasetConfig,
    dataset_config,
    run_gcv,
)
from evaluation.predictors.nn_base import load_cache


def add_common_args(
    p: argparse.ArgumentParser,
    *,
    method: str,
    cache_arg: str,
    require_arg: str,
    hitrate_arg: str,
    predictor_choices: list[str],
    cache_ext: str = "pkl",
) -> argparse.ArgumentParser:
    """Add the argparse arguments shared between the MINT and PPLM GCV scripts."""
    flag = cache_arg.replace("_cache", "").replace("_", "-")
    p.add_argument(
        "--dataset",
        required=True,
        type=dataset_arg,
        choices=list(DATASET_CONFIGS),
        help="Dataset configuration to use",
    )
    p.add_argument(
        "--predictor",
        default="seq_diff",
        choices=predictor_choices,
        help=f"{method} predictor variant (default: seq_diff)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for MLP training (default: 42)",
    )
    p.add_argument(
        "--n-gcv", type=int, default=30,
        help="Number of GCV iterations (default: 30)",
    )
    p.add_argument(
        "--outdir", default=str(GCV_RESULTS_DIR),
        help="Output directory for results (default: results/gcv/)",
    )
    p.add_argument(
        f"--{flag}-cache",
        default="",
        dest=cache_arg,
        help=f"Path to {method} embedding cache .{cache_ext} "
             f"(default: the canonical per-dataset cache, else the built-in "
             f"CACHE_PATH). Generate with precompute_{flag}_embeddings.py.",
    )
    p.add_argument(
        f"--min-{flag}-hit-rate",
        type=float,
        default=1.0,
        dest=hitrate_arg,
        help=f"Minimum MUT key hit rate before --require-{flag} aborts "
             f"(default: 1.0, i.e. every key must be present -- all methods "
             f"score the same rows).",
    )
    p.add_argument(
        f"--require-{flag}",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest=require_arg,
        help=f"Abort if the {method} cache is missing or its hit rate is below "
             f"--min-{flag}-hit-rate (default: True). Pass --no-require-{flag} "
             f"to run with missing-key rows falling back to prior.",
    )
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from an existing checkpoint, continuing after the last "
             "completed GCV seed (default: True).",
    )
    return p


def _rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / max(total, 1):.1%})"


def audit_embedding_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    *,
    method: str,
    cache_path: str,
    wt_key: Callable[[str, str], str],
    mut_key: Callable[[str, str, str], str],
    is_present: Callable[[Any, str], bool],
    min_hit_rate: float,
    require: bool,
    precompute_script: str,
    require_flag: str,
    cache_flag: str,
    entry_noun: str,
    extra_report: Callable[[Any, set, set], None] | None = None,
) -> None:
    """Load an embedding cache and report WT/MUT key hit rates.

    Aborts when require=True and either a WT key is missing or the mutant hit
    rate falls below min_hit_rate. Missing WT keys mean the seq-diff predictor
    sees all-zero embeddings; for MINT, missing res keys additionally cripple
    site_diff (reported via extra_report).
    """
    print(f"\n── {method} cache audit ──", flush=True)
    print(f"path: {cache_path}", flush=True)

    cache = load_cache(cache_path)
    if cache is None:
        msg = (
            f"{method} cache not found at {cache_path}. "
            f"Run {precompute_script} --dataset {cfg.name} "
            f"to generate it, then pass the path via {cache_flag}."
        )
        if require:
            raise SystemExit("ABORT: " + msg + f"\nPass {require_flag} to override.")
        print("WARNING: " + msg, flush=True)
        return

    print(f"entries: {len(cache)}", flush=True)

    interactors = ordered_df["interactor"].astype(str)
    partners    = ordered_df["partner"].astype(str)
    mutations   = ordered_df["mutation"].astype(str)

    unique_pairs = set(zip(interactors, partners))
    unique_muts  = set(zip(interactors, partners, mutations))

    wt_hits  = sum(1 for a, b    in unique_pairs if is_present(cache, wt_key(a, b)))
    mut_hits = sum(1 for a, b, m in unique_muts  if is_present(cache, mut_key(a, b, m)))
    print(f"WT  {entry_noun}: {_rate(wt_hits,  len(unique_pairs))}", flush=True)
    print(f"MUT {entry_noun}: {_rate(mut_hits, len(unique_muts))}", flush=True)

    if extra_report is not None:
        extra_report(cache, unique_pairs, unique_muts)

    mut_rate = mut_hits / max(len(unique_muts), 1)
    if wt_hits < len(unique_pairs) or mut_rate < min_hit_rate:
        missing_wt = sorted(wt_key(a, b) for a, b in unique_pairs
                            if not is_present(cache, wt_key(a, b)))
        missing_mut = sorted(mut_key(a, b, m) for a, b, m in unique_muts
                             if not is_present(cache, mut_key(a, b, m)))
        msg = (
            f"{method} cache is incomplete: {len(missing_wt)} WT and "
            f"{len(missing_mut)} mutant {entry_noun} missing "
            f"(mutant hit rate {mut_rate:.1%}, required {min_hit_rate:.1%}). "
            f"Every method is scored on the same rows, so a partial cache is an "
            f"error. Run {precompute_script} --dataset {cfg.name} "
            f"to fill it.\n  missing WT : {missing_wt[:10]}"
            f"\n  missing MUT: {missing_mut[:10]}"
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


def run_seq_site_gcv(
    args: argparse.Namespace,
    *,
    method: str,
    predictor_map: dict[str, Any],
    cache_module: Any,
    predictor_classes: list[Any],
    cache_arg: str,
    audit_fn: Callable,
    require_arg: str,
    hitrate_arg: str,
    canonical_ext: str = "pkl",
) -> None:
    """Shared run() body for the MINT and PPLM GCV scripts.

    Wires the cache path from --<method>-cache or from the canonical
    per-dataset cache, then calls run_gcv() with audit_fn as the preflight hook.
    """
    cache_path: str = getattr(args, cache_arg)
    if cache_path:
        cache_module.CACHE_PATH = cache_path
        for cls in predictor_classes:
            cls._cache_path = cache_path
        print(f"{method} cache path overridden: {cache_path}", flush=True)
    else:
        canonical = TRAINING_EVAL_DIR / f"{args.dataset}_{method.lower()}.{canonical_ext}"
        if canonical.exists():
            cache_module.CACHE_PATH = str(canonical)
            for cls in predictor_classes:
                cls._cache_path = str(canonical)
            print(f"{method} cache: {canonical}", flush=True)

    cfg    = dataset_config(args.dataset)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    PredictorClass = predictor_map[args.predictor]
    print(f"Predictor: {PredictorClass().name}", flush=True)

    run_gcv(
        cfg, args,
        result_stem=f"{method}_{args.predictor}_{cfg.name}",
        make_predictor=lambda a: PredictorClass(seed=a.seed),
        preflight=lambda odf, c_, a: audit_fn(
            odf, c_, a.predictor,
            getattr(a, hitrate_arg),
            getattr(a, require_arg),
        ),
    )
