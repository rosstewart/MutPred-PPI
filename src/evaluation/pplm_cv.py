#!/usr/bin/env python
"""PPLM seq-diff / site-diff MLP group cross-validation training script.

Uses the PPLM embedding-based MLP predictors (PPLMSeqDiff / PPLMSiteDiff from
predictors/pplm_mlp.py). Both are sklearn-style (no GPU at inference):

  pplm_seq_diff  — mean(mut_A - wt_A) | mean(wt_B)  →  small MLP
  pplm_site_diff — emb_mut_A[site] - emb_wt_A[site] | mean(wt_B)  →  small MLP

The argparse skeleton, cache audit and GCV driver are shared with mint_cv.py via
evaluation/embedding_cv_common.py; only the cache key schema differs. PPLM
stores raw per-residue matrices for both chains under one key, because embed_B
differs between WT and MUT — chain A's substituted token reaches chain B through
cross-attention — so the partner mean is recomputed per variant rather than
shared across the pair.

PPLM embeddings must be precomputed before running this script; use
precompute_pplm_embeddings.py to generate the cache.

Usage:
    conda run -n ppi python pplm_cv.py --dataset sahni_fragoza \\
        --pplm-cache $MUTPRED_CACHE_DIR/pplm_cache.pkl

    conda run -n ppi python pplm_cv.py \\
        --dataset sahni_fragoza_mapped090826 \\
        --predictor site_diff \\
        --pplm-cache /path/to/cache.pkl \\
        --n-gcv 30 --outdir /path/to/results/
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from utils.gcv_common import DatasetConfig
import evaluation.predictors.pplm_mlp as _pplm_mod
from evaluation.predictors.pplm_mlp import PPLMSeqDiff, PPLMSiteDiff
from evaluation.embedding_cv_common import (
    add_common_args,
    audit_embedding_cache,
    run_seq_site_gcv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_PREDICTOR_MAP = {
    "seq_diff":  PPLMSeqDiff,
    "site_diff": PPLMSiteDiff,
}


def _has_embeds(cache, key: str) -> bool:
    """True when the entry exists and carries both chain embeddings.

    The cache predates this pipeline and can hold entries written by other PPLM
    experiments that only stored ppi_wt / affinity_wt, so presence of the key is
    not enough — embed_A and embed_B must both be there.
    """
    entry = cache.get(key)
    return (entry is not None
            and entry.get("embed_A") is not None
            and entry.get("embed_B") is not None)


def audit_pplm_cache(
    ordered_df: pd.DataFrame,
    cfg: DatasetConfig,
    predictor: str,
    min_hit_rate: float,
    require: bool,
) -> None:
    """Audit the PPLM cache, requiring embed_A/embed_B on every entry."""
    audit_embedding_cache(
        ordered_df, cfg,
        method="PPLM",
        cache_path=_pplm_mod.CACHE_PATH,
        wt_key=lambda a, b: f"{a}_{b}",
        mut_key=lambda a, b, m: f"{a}_{b}_{m}",
        is_present=_has_embeds,
        min_hit_rate=min_hit_rate,
        require=require,
        precompute_script="precompute_pplm_embeddings.py",
        require_flag="--no-require-pplm",
        cache_flag="--pplm-cache",
        entry_noun="entries with embeds",
    )


def run(args: argparse.Namespace) -> None:
    run_seq_site_gcv(
        args,
        method="PPLM",
        predictor_map=_PREDICTOR_MAP,
        cache_module=_pplm_mod,
        predictor_classes=[PPLMSeqDiff, PPLMSiteDiff],
        cache_arg="pplm_cache",
        audit_fn=audit_pplm_cache,
        require_arg="require_pplm",
        hitrate_arg="min_pplm_hit_rate",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPLM MLP GCV training (mirrors SWING / eSIG-Net / MINT pipeline)"
    )
    add_common_args(
        p,
        method="PPLM",
        cache_arg="pplm_cache",
        require_arg="require_pplm",
        hitrate_arg="min_pplm_hit_rate",
        predictor_choices=list(_PREDICTOR_MAP),
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
