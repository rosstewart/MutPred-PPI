"""PPLM embedding-based MLP predictors.

Cache: $MUTPRED_DATA_ROOT/2026/pplm_cache.pkl

  "{id_a}_{id_b}"            → WT entry:
      mean:    (1280,) float32      full-pair mean (all La+Lb residues)
      embed_A: (La, 1280) float16   per-residue chain-A embeddings
      embed_B: (Lb, 1280) float16   per-residue chain-B embeddings
  "{id_a}_{id_b}_{var}" → MUT entry:
      mean:    (1280,) float32
      embed_A: (La, 1280) float16
      embed_B: (Lb, 1280) float16

Keys use ZERO-BASED variant positions (e.g. 'E79K' for 1-based 'E80K').

Two predictors:
  pplm_seq_diff  — mean_pool_all(mut_pair) - mean_pool_all(wt_pair)  →  1280-dim  →  MLP
  pplm_site_diff — emb_mut_a[site] - emb_wt_a[site]                 →  1280-dim  →  MLP
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

from pathlib import Path as _Path

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import cache_file  # noqa: E402

CACHE_PATH = str(cache_file("pplm_cache.pkl"))


def _pplm_keys(row: pd.Series):
    """Return (wt_key, mut_key) for this row."""
    id_a = row["interactor"]
    id_b = row["partner"]
    var = str(row["mutation"])          # 1-based, as stored in the tables
    return f"{id_a}_{id_b}", f"{id_a}_{id_b}_{var}"


def _get_embeds(cache: dict, key: str):
    """Return (embed_A, embed_B) arrays or (None, None) on miss."""
    entry = cache.get(key)
    if entry is None:
        return None, None
    ea = entry.get("embed_A")
    eb = entry.get("embed_B")
    return ea, eb


def _lookup_pplm_seq(cache: dict, row: pd.Series) -> Optional[np.ndarray]:
    """Seq diff: mean_mut - mean_wt  →  1280-dim (precomputed full-pair means)."""
    wt_key, mut_key = _pplm_keys(row)
    wt_entry  = cache.get(wt_key)
    mut_entry = cache.get(mut_key)
    if wt_entry is None or mut_entry is None:
        return None
    mean_wt  = wt_entry.get("mean")
    mean_mut = mut_entry.get("mean")
    if mean_wt is None or mean_mut is None:
        return None
    return mean_mut.astype(np.float32) - mean_wt.astype(np.float32)


def _lookup_pplm_site(cache: dict, row: pd.Series) -> Optional[np.ndarray]:
    """Site diff: mut_A[site] - wt_A[site]  →  1280-dim."""
    wt_key, mut_key = _pplm_keys(row)
    _, pos0, _ = parse_mutation(row["mutation"])

    wt_a, _ = _get_embeds(cache, wt_key)
    mut_a, _ = _get_embeds(cache, mut_key)
    if wt_a is None or mut_a is None:
        return None

    wt_a32  = wt_a.astype(np.float32)
    mut_a32 = mut_a.astype(np.float32)
    if pos0 >= len(wt_a32) or pos0 >= len(mut_a32):
        return None

    return mut_a32[pos0] - wt_a32[pos0]


@register
class PPLMSeqDiff(CacheMLPPredictor):
    """PPLM MLP: sequence-level mean diff (mut - wt) + partner."""

    _cache_path = CACHE_PATH

    def _extract_features(self, row: pd.Series) -> Optional[np.ndarray]:
        cache = self._get_cache()
        return None if cache is None else _lookup_pplm_seq(cache, row)

    @property
    def name(self) -> str:
        return "pplm_seq_diff"


@register
class PPLMSiteDiff(CacheMLPPredictor):
    """PPLM MLP: mutation-site embedding diff + partner."""

    _cache_path = CACHE_PATH

    def _extract_features(self, row: pd.Series) -> Optional[np.ndarray]:
        cache = self._get_cache()
        return None if cache is None else _lookup_pplm_site(cache, row)

    @property
    def name(self) -> str:
        return "pplm_site_diff"


# ── PCA variants ─────────────────────────────────────────────────────────────

for _mode, _base in [("seq_diff", PPLMSeqDiff), ("site_diff", PPLMSiteDiff)]:
    for _k in (16, 32, 64):
        _cls = type(
            f"PPLM{_mode.replace('_','').title()}PCA{_k}",
            (_base,),
            {"_n_pca": _k, "name": property(lambda self, m=_mode, k=_k: f"pplm_{m}_pca{k}")},
        )
        register(_cls)
