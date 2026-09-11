#!/usr/bin/env python
"""MINT seq-diff / site-diff MLP group cross-validation training script.

Uses the MINT embedding-based MLP predictors (MINTSeqDiff / MINTSiteDiff from
predictors/mint_mlp.py). Both are sklearn-style (no GPU required at inference):

  mint_seq_diff  — mean(mut_A - wt_A) | mean_partner  →  small MLP
  mint_site_diff — emb_mut_A[site] - emb_wt_A[site] | mean_partner  →  small MLP

The argparse skeleton, cache audit and GCV driver are shared with pplm_cv.py via
evaluation/embedding_cv_common.py; only the cache key schema differs. MINT
stores pre-reduced arrays under flat string keys — `mean_{a}_{b}` is already
pooled over all La+Lb residues — with per-residue chain A held separately under
`res_*` for site_diff.

MINT embeddings must be precomputed before running this script; use
precompute_mint_embeddings.py to generate the cache.

Usage:
    conda run -n ppi python mint_cv.py --dataset sahni_fragoza \\
        --mint-cache $MUTPRED_DATA_ROOT/2026/mint_cache/sahni_fragoza.pkl

    conda run -n ppi python mint_cv.py \\
        --dataset sahni_fragoza_mapped090826 \\
        --predictor site_diff \\
        --mint-cache /path/to/cache.pkl \\
        --n-gcv 30 \\
        --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from utils.gcv_common import DatasetConfig
import evaluation.predictors.mint_mlp as _mint_mod
from evaluation.predictors.mint_mlp import MINTSeqDiff, MINTSiteDiff
from evaluation.embedding_cv_common import (
    add_common_args,
    audit_embedding_cache,
    run_seq_site_gcv,
    _rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PREDICTOR_MAP = {
    "seq_diff":  MINTSeqDiff,
    "site_diff": MINTSiteDiff,
}


def _wt_key(a: str, b: str) -> str:
    return f"mean_{a}_{b}"


def _mut_key(a: str, b: str, m: str) -> str:
    return f"mean_{a}_{b}_{m}"


def audit_mint_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    predictor: str,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Audit the MINT cache; for site_diff also report per-residue key coverage."""

    def _res_report(cache, unique_pairs, unique_muts) -> None:
        if predictor != "site_diff":
            return
        res_wt_hits  = sum(1 for a, b    in unique_pairs
                           if f"res_wt_pair_{a}_{b}" in cache)
        res_mut_hits = sum(1 for a, b, m in unique_muts
                           if f"res_mut_pair_{m}_{a}_{b}" in cache)
        print(f"res WT  keys: {_rate(res_wt_hits,  len(unique_pairs))}", flush=True)
        print(f"res MUT keys: {_rate(res_mut_hits, len(unique_muts))}", flush=True)

    audit_embedding_cache(
        ordered_df, cfg,
        method="MINT",
        cache_path=_mint_mod.CACHE_PATH,
        wt_key=_wt_key,
        mut_key=_mut_key,
        is_present=lambda cache, key: key in cache,
        min_hit_rate=min_hit_rate,
        require=require,
        precompute_script="precompute_mint_embeddings.py",
        require_flag="--no-require-mint",
        cache_flag="--mint-cache",
        entry_noun="mean keys",
        extra_report=_res_report,
    )


def run(args: argparse.Namespace) -> None:
    run_seq_site_gcv(
        args,
        method="MINT",
        predictor_map=_PREDICTOR_MAP,
        cache_module=_mint_mod,
        predictor_classes=[MINTSeqDiff, MINTSiteDiff],
        cache_arg="mint_cache",
        audit_fn=audit_mint_cache,
        require_arg="require_mint",
        hitrate_arg="min_mint_hit_rate",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MINT MLP GCV training (mirrors SWING / eSIG-Net pipeline)"
    )
    add_common_args(
        p,
        method="MINT",
        cache_arg="mint_cache",
        require_arg="require_mint",
        hitrate_arg="min_mint_hit_rate",
        predictor_choices=list(_PREDICTOR_MAP),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
