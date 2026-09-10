#!/usr/bin/env python
"""MINT seq-diff / site-diff MLP group cross-validation training script.

Mirrors esignet_gcv_iter.py but uses the MINT embedding-based MLP predictors
(MINTSeqDiff / MINTSiteDiff from predictors/mint_mlp.py) instead of eSIG-Net.

Both predictors are sklearn-style (no GPU required at inference):
  mint_seq_diff  — mean(mut_A - wt_A) | mean_partner  →  small MLP
  mint_site_diff — emb_mut_A[site] - emb_wt_A[site] | mean_partner  →  small MLP

MINT embeddings must be precomputed before running this script; use
precompute_mint_embeddings.py to generate the cache.

Usage:
    conda run -n ppi python mint_gcv_iter.py --dataset sahni_fragoza \\
        --mint-cache $MUTPRED_DATA_ROOT/2026/mint_cache/sahni_fragoza.pkl

    conda run -n ppi python mint_gcv_iter.py \\
        --dataset sahni_fragoza_mapped090826 \\
        --predictor site_diff \\
        --mint-cache /path/to/cache.pkl \\
        --n-gcv 30 \\
        --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── shared code (see src/evaluation/esignet_cv.py and
#    src/evaluation/predictors/) ───────────────────────────────────────────────
from paths import DATASETS_DIR, GCV_RESULTS_DIR, TRAINING_EVAL_DIR  # noqa: E402
# Shared GCV data-loading layer (see src/utils/gcv_common.py).
from utils.gcv_common import DATASET_CONFIGS, DatasetConfig, run_gcv

import evaluation.predictors.mint_mlp as _mint_mod   # noqa: E402
from evaluation.predictors.mint_mlp import MINTSeqDiff, MINTSiteDiff  # noqa: E402
from evaluation.predictors.nn_base import load_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PREDICTOR_MAP = {
    "seq_diff":  MINTSeqDiff,
    "site_diff": MINTSiteDiff,
}


# ── MINT cache audit ──────────────────────────────────────────────────────────

def audit_mint_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    predictor: str,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Load the MINT cache and report mean/res key hit rates.

    Aborts when require=True and the mutant mean-key hit rate falls below
    min_hit_rate.  Missing mean keys mean MINTSeqDiff sees all-zero embeddings;
    missing res keys additionally cripple MINTSiteDiff.
    """
    path = _mint_mod.CACHE_PATH
    print(f"\n── MINT cache audit ──", flush=True)
    print(f"path: {path}", flush=True)

    cache = load_cache(path)
    if cache is None:
        msg = (
            f"MINT cache not found at {path}. "
            f"Run precompute_mint_embeddings.py --dataset {cfg.name} "
            f"to generate it, then pass the path via --mint-cache."
        )
        if require:
            raise SystemExit("ABORT: " + msg + "\nPass --no-require-mint to override.")
        print("WARNING: " + msg, flush=True)
        return

    print(f"entries: {len(cache)}", flush=True)

    # Filter out NaN rows (unmatched vt_ids produce NaN mutations)
    valid_df    = ordered_df
    interactors = valid_df["interactor"].astype(str)
    partners    = valid_df["partner"].astype(str)
    mutations   = valid_df["mutation"].astype(str)

    unique_pairs = set(zip(interactors, partners))
    unique_muts  = set(zip(interactors, partners, mutations))

    def _rate(hits, total):
        return f"{hits}/{total} ({hits / max(total, 1):.1%})"

    mean_wt_hits  = sum(1 for a, b    in unique_pairs if f"mean_{a}_{b}" in cache)
    mean_mut_hits = sum(1 for a, b, m in unique_muts
                        if f"mean_{a}_{b}_{m}" in cache)
    print(f"mean WT  keys: {_rate(mean_wt_hits,  len(unique_pairs))}", flush=True)
    print(f"mean MUT keys: {_rate(mean_mut_hits, len(unique_muts))}", flush=True)

    if predictor == "site_diff":
        res_wt_hits  = sum(1 for a, b    in unique_pairs if f"res_wt_pair_{a}_{b}" in cache)
        res_mut_hits = sum(1 for a, b, m in unique_muts
                           if f"res_mut_pair_{m}_{a}_{b}" in cache)
        print(f"res WT  keys: {_rate(res_wt_hits,  len(unique_pairs))}", flush=True)
        print(f"res MUT keys: {_rate(res_mut_hits, len(unique_muts))}", flush=True)

    mut_rate = mean_mut_hits / max(len(unique_muts), 1)
    if mean_wt_hits < len(unique_pairs) or mut_rate < min_hit_rate:
        missing_wt = sorted(f"mean_{a}_{b}" for a, b in unique_pairs
                            if f"mean_{a}_{b}" not in cache)
        missing_mut = sorted(f"mean_{a}_{b}_{m}" for a, b, m in unique_muts
                             if f"mean_{a}_{b}_{m}" not in cache)
        msg = (
            f"MINT cache is incomplete: {len(missing_wt)} WT and "
            f"{len(missing_mut)} mutant mean-keys missing "
            f"(mutant hit rate {mut_rate:.1%}, required {min_hit_rate:.1%}). "
            f"Every method is scored on the same rows, so a partial cache is an "
            f"error. Run precompute_mint_embeddings.py --dataset {cfg.name} "
            f"to fill it.\n  missing WT : {missing_wt[:10]}"
            f"\n  missing MUT: {missing_mut[:10]}"
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


# ── main GCV loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.mint_cache:
        _mint_mod.CACHE_PATH = args.mint_cache
        MINTSeqDiff._cache_path  = args.mint_cache
        MINTSiteDiff._cache_path = args.mint_cache
        print(f"MINT cache path overridden: {args.mint_cache}", flush=True)
    else:
        canonical = TRAINING_EVAL_DIR / f"{args.dataset}_mint.pkl"
        if canonical.exists():
            _mint_mod.CACHE_PATH = str(canonical)
            MINTSeqDiff._cache_path  = str(canonical)
            MINTSiteDiff._cache_path = str(canonical)
            print(f"MINT cache: {canonical}", flush=True)

    cfg       = DATASET_CONFIGS[args.dataset]
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    PredictorClass = _PREDICTOR_MAP[args.predictor]
    print(f"Predictor: {PredictorClass().name}", flush=True)

    run_gcv(
        cfg, args,
        result_stem=f"MINT_{args.predictor}_{cfg.name}",
        make_predictor=lambda a: PredictorClass(seed=a.seed),
        preflight=lambda odf, c_, a: audit_mint_cache(
            odf, c_, a.predictor, a.min_mint_hit_rate, a.require_mint),
    )

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MINT MLP GCV training (mirrors SWING / eSIG-Net pipeline)"
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS),
        help="Dataset configuration to use",
    )
    p.add_argument(
        "--predictor",
        default="seq_diff",
        choices=list(_PREDICTOR_MAP),
        help="MINT predictor variant (default: seq_diff)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MLP training (default: 42)",
    )
    p.add_argument(
        "--n-gcv",
        type=int,
        default=30,
        help="Number of GCV iterations (default: 30)",
    )
    p.add_argument(
        "--outdir",
        default=str(GCV_RESULTS_DIR),
        help="Output directory for results (default: CV splits dir)",
    )
    p.add_argument(
        "--mint-cache",
        default="",
        help=(
            "Path to MINT embedding cache .pkl "
            "(default: built-in CACHE_PATH in mint_mlp.py). "
            "Generate with precompute_mint_embeddings.py."
        ),
    )
    p.add_argument(
        "--min-mint-hit-rate",
        type=float,
        default=1.0,
        help="Minimum MUT key hit rate before --require-mint aborts (default: 1.0, "
             "i.e. every key must be present -- all methods score the same rows).",
    )
    p.add_argument(
        "--require-mint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Abort if the MINT cache is missing or has hit rate below "
            "--min-mint-hit-rate (default: True). "
            "Pass --no-require-mint to run with missing-key rows falling back to prior."
        ),
    )
    p.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Resume from an existing checkpoint, continuing after the last "
             "completed GCV seed (default: True).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
