"""MINT embedding-based MLP predictors.

Cache: $MUTPRED_DATA_ROOT/2026/mint_cache/mint_cache_v2.pkl

Keys use ZERO-BASED variant positions (e.g. 'E79K' for 1-based 'E80K').

Full-pair mean embeddings (1280-dim, sep_chains=False — mean over all La+Lb residues):
  "mean_{id_a}_{id_b}"              → WT full-pair mean
  "mean_{id_a}_{id_b}_{var}"   → MUT full-pair mean

Per-residue chain-A embeddings (La, 1280):
  "res_wt_pair_{id_a}_{id_b}"              → WT chain A in pair context
  "res_mut_pair_{var}_{id_a}_{id_b}"  → MUT chain A in pair context

Two predictors:
  mint_seq_diff  — mean_pool(mut_pair) - mean_pool(wt_pair)  →  1280-dim  →  MLP
  mint_site_diff — emb_mut_a[site] - emb_wt_a[site]        →  1280-dim  →  MLP
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import register
from .nn_base import (
    CacheMLPPredictor,
    parse_mutation,
)

logger = logging.getLogger(__name__)


# --- repo-relative path resolution (see src/paths.py) ---
from paths import cache_file  # noqa: E402

CACHE_PATH = str(cache_file("mint_cache.pkl"))


def _mint_keys(row: pd.Series):
    """Return (wt_mean_key, mut_mean_key, wt_res_key, mut_res_key) for a row."""
    id_a = row["interactor"]
    id_b = row["partner"]
    var = str(row["mutation"])          # 1-based, as stored in the tables
    return (
        f"mean_{id_a}_{id_b}",
        f"mean_{id_a}_{id_b}_{var}",
        f"res_wt_pair_{id_a}_{id_b}",
        f"res_mut_pair_{var}_{id_a}_{id_b}",
    )


def _lookup_mint_seq(cache: dict, row: pd.Series) -> Optional[np.ndarray]:
    """
    Seq diff: mean_pool_all_residues(mut_pair) - mean_pool_all_residues(wt_pair)

    Follows MINT's sep_chains=False path (embeddings_mint.py): mean over ALL
    La+Lb residues → 1280-dim per pair. Feature = mut_mean - wt_mean (1280-dim).
    """
    k_wt_mean, k_mut_mean, _, _ = _mint_keys(row)
    mean_wt  = cache.get(k_wt_mean)
    mean_mut = cache.get(k_mut_mean)
    if mean_wt is None or mean_mut is None:
        return None
    return mean_mut.astype(np.float32) - mean_wt.astype(np.float32)


def _lookup_mint_site(cache: dict, row: pd.Series) -> Optional[np.ndarray]:
    """Site diff: emb_mut_a[site] - emb_wt_a[site]  →  1280-dim."""
    _, _, k_wt_res, k_mut_res = _mint_keys(row)
    _, pos0, _ = parse_mutation(row["mutation"])

    wt_res  = cache.get(k_wt_res)
    mut_res = cache.get(k_mut_res)
    if wt_res is None or mut_res is None:
        return None

    wt_res  = wt_res.astype(np.float32)
    mut_res = mut_res.astype(np.float32)
    if pos0 >= len(wt_res) or pos0 >= len(mut_res):
        return None

    return mut_res[pos0] - wt_res[pos0]


@register
class MINTSeqDiff(CacheMLPPredictor):
    """MINT MLP: sequence-level mean diff (mut - wt) + partner context."""

    _cache_path = CACHE_PATH

    def _extract_features(self, row: pd.Series) -> Optional[np.ndarray]:
        cache = self._get_cache()
        return None if cache is None else _lookup_mint_seq(cache, row)

    @property
    def name(self) -> str:
        return "mint_seq_diff"


@register
class MINTSiteDiff(CacheMLPPredictor):
    """MINT MLP: mutation-site embedding diff + partner."""

    _cache_path = CACHE_PATH

    def _extract_features(self, row: pd.Series) -> Optional[np.ndarray]:
        cache = self._get_cache()
        return None if cache is None else _lookup_mint_site(cache, row)

    @property
    def name(self) -> str:
        return "mint_site_diff"


# ── PCA variants ─────────────────────────────────────────────────────────────

for _mode, _base in [("seq_diff", MINTSeqDiff), ("site_diff", MINTSiteDiff)]:
    for _k in (16, 32, 64):
        _cls = type(
            f"MINT{_mode.replace('_','').title()}PCA{_k}",
            (_base,),
            {"_n_pca": _k, "name": property(lambda self, m=_mode, k=_k: f"mint_{m}_pca{k}")},
        )
        register(_cls)
