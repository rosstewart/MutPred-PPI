#!/usr/bin/env python
"""PPLM seq-diff / site-diff MLP group cross-validation training script.

Mirrors mint_gcv_iter.py but uses PPLM embedding-based MLP predictors
(PPLMSeqDiff / PPLMSiteDiff from predictors/pplm_mlp.py).

Both predictors are sklearn-style (no GPU at inference):
  pplm_seq_diff  — mean(mut_A - wt_A) | mean(wt_B)  →  small MLP
  pplm_site_diff — emb_mut_A[site] - emb_wt_A[site] | mean(wt_B)  →  small MLP

PPLM embeddings must be precomputed before running this script; use
precompute_pplm_embeddings.py to generate the cache.

Usage:
    conda run -n ppi python pplm_gcv_iter.py --dataset sahni_fragoza \\
        --pplm-cache $MUTPRED_CACHE_DIR/pplm_cache.pkl

    conda run -n ppi python pplm_gcv_iter.py \\
        --dataset sahni_fragoza_varchamp1p_cava \\
        --predictor site_diff \\
        --pplm-cache /path/to/cache.pkl \\
        --n-gcv 30 --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# ── shared code (see src/evaluation/esignet_cv.py and
#    src/evaluation/predictors/) ───────────────────────────────────────────────
from paths import DATASETS_DIR, GCV_RESULTS_DIR  # noqa: E402
# Shared GCV data-loading layer (see src/evaluation/gcv_common.py).
from evaluation.gcv_common import DATASET_CONFIGS, DatasetConfig, run_gcv

import evaluation.predictors.pplm_mlp as _pplm_mod   # noqa: E402
from evaluation.predictors.pplm_mlp import PPLMSeqDiff, PPLMSiteDiff  # noqa: E402
from evaluation.predictors.nn_base import load_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PREDICTOR_MAP = {
    "seq_diff":  PPLMSeqDiff,
    "site_diff": PPLMSiteDiff,
}


# ── PPLM cache audit ──────────────────────────────────────────────────────────

def audit_pplm_cache(
    ordered_df,
    cfg: DatasetConfig,
    predictor: str,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Load the PPLM cache and report WT/MUT entry hit rates.

    Checks that cache entries have embed_A and embed_B sub-keys, since the
    existing pplm_cache.pkl may have entries from other pipelines that lack
    embeddings (only ppi_wt / affinity_wt etc.).
    """
    path = _pplm_mod.CACHE_PATH
    print(f"\n── PPLM cache audit ──", flush=True)
    print(f"path: {path}", flush=True)

    cache = load_cache(path)
    if cache is None:
        msg = (
            f"PPLM cache not found at {path}. "
            f"Run precompute_pplm_embeddings.py --dataset {cfg.name} "
            f"to generate it, then pass the path via --pplm-cache."
        )
        if require:
            raise SystemExit("ABORT: " + msg + "\nPass --no-require-pplm to override.")
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

    def _has_embeds(key):
        entry = cache.get(key)
        return (entry is not None
                and entry.get("embed_A") is not None
                and entry.get("embed_B") is not None)

    def _rate(hits, total):
        return f"{hits}/{total} ({hits / max(total, 1):.1%})"

    wt_hits  = sum(1 for a, b    in unique_pairs if _has_embeds(f"{a}_{b}"))
    mut_hits = sum(1 for a, b, m in unique_muts
                   if _has_embeds(f"{a}_{b}_{m}"))

    print(f"WT  entries with embeds: {_rate(wt_hits,  len(unique_pairs))}", flush=True)
    print(f"MUT entries with embeds: {_rate(mut_hits, len(unique_muts))}", flush=True)

    mut_rate = mut_hits / max(len(unique_muts), 1)
    if wt_hits < len(unique_pairs) or mut_rate < min_hit_rate:
        missing_wt = sorted(f"{a}_{b}" for a, b in unique_pairs
                            if not _has_embeds(f"{a}_{b}"))
        missing_mut = sorted(f"{a}_{b}_{m}" for a, b, m in unique_muts
                             if not _has_embeds(f"{a}_{b}_{m}"))
        msg = (
            f"PPLM cache is incomplete: {len(missing_wt)} WT and "
            f"{len(missing_mut)} mutant entries missing embeddings "
            f"(mutant hit rate {mut_rate:.1%}, required {min_hit_rate:.1%}). "
            f"Every method is scored on the same rows, so a partial cache is an "
            f"error. Run precompute_pplm_embeddings.py --dataset {cfg.name} "
            f"to fill it.\n  missing WT : {missing_wt[:10]}"
            f"\n  missing MUT: {missing_mut[:10]}"
        )
        if require:
            raise SystemExit("ABORT: " + msg)
        print("WARNING: " + msg, flush=True)


# ── main GCV loop ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    if args.pplm_cache:
        _pplm_mod.CACHE_PATH     = args.pplm_cache
        PPLMSeqDiff._cache_path  = args.pplm_cache
        PPLMSiteDiff._cache_path = args.pplm_cache
        print(f"PPLM cache path overridden: {args.pplm_cache}", flush=True)
    else:
        canonical = DATASETS_DIR / "mapped090826" / f"{args.dataset}_pplm.pkl"
        if canonical.exists():
            _pplm_mod.CACHE_PATH     = str(canonical)
            PPLMSeqDiff._cache_path  = str(canonical)
            PPLMSiteDiff._cache_path = str(canonical)
            print(f"PPLM cache: {canonical}", flush=True)

    cfg       = DATASET_CONFIGS[args.dataset]
    outdir    = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    PredictorClass = _PREDICTOR_MAP[args.predictor]
    print(f"Predictor: {PredictorClass().name}", flush=True)

    run_gcv(
        cfg, args,
        result_stem=f"PPLM_{args.predictor}_{cfg.name}",
        make_predictor=lambda a: PredictorClass(seed=a.seed),
        preflight=lambda odf, c_, a: audit_pplm_cache(
            odf, c_, a.predictor, a.min_pplm_hit_rate, a.require_pplm),
    )

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPLM MLP GCV training (mirrors SWING / eSIG-Net / MINT pipeline)"
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
        help="PPLM predictor variant (default: seq_diff)",
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
        "--pplm-cache",
        default="",
        help=(
            "Path to PPLM embedding cache .pkl "
            "(default: built-in CACHE_PATH in pplm_mlp.py). "
            "Generate with precompute_pplm_embeddings.py."
        ),
    )
    p.add_argument(
        "--min-pplm-hit-rate",
        type=float,
        default=1.0,
        help="Minimum MUT embed hit rate before --require-pplm aborts (default: 1.0, "
             "i.e. every key must be present -- all methods score the same rows).",
    )
    p.add_argument(
        "--require-pplm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Abort if the PPLM cache is missing or MUT hit rate is below "
            "--min-pplm-hit-rate (default: True). "
            "Pass --no-require-pplm to run with cache-miss rows falling back to prior."
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
