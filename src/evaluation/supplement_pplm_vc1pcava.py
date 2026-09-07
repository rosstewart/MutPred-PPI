#!/usr/bin/env python
"""PPLM supplemental blind test on VC1p + CAVA entries.

These entries are in the SFVCFP pkl but were excluded from the main VCFP blind
test because their graphs use gene-name/Entrez IDs rather than UniProt IDs.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl (same mapping
used in load_sahni_fragoza_varchamp1p_cava() for GCV construction). C1/C2/C3
classification uses the remapped UniProt IDs against the SF protein set.

The PPLM GCV cache already covers all vc1pcava proteins (confirmed by precompute
showing 0 new embeddings needed), so no supplemental precompute is required.

Output vt_ids use canonical 0-based format: "UNIPROT1 UNIPROT2 MUT_0based"
(matching the MutPred-PPI vc1pcava supplement, so restratify intersection works).

22 entries shared between VC1p and CAVA are deduplicated.

`run()` is importable (e.g. from src/evaluation/run_vcfp_blind_test.py) and
returns the method description string used for the saved npy files.

Usage:
    conda run -n ppi python supplement_pplm_vc1pcava.py [--predictor seq_diff|site_diff]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_CV_MOD = Path(__file__).resolve().parent  # src/evaluation — also holds predictors/ now
sys.path.insert(0, str(_CV_MOD))

import predictors.pplm_mlp as _pplm_mod  # noqa: E402
from predictors.pplm_mlp import PPLMSeqDiff, PPLMSiteDiff  # noqa: E402
from predictors import nn_base  # noqa: E402

from vcfp_common import (  # noqa: E402
    build_sf_proteins, load_sf_train_df, load_vc1pcava_sources,
    iter_vc1pcava_entries, save_vc1pcava_supplement,
)


# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import cache_file  # noqa: E402

_GCV_CACHE = cache_file("pplm_cache.pkl")

_DESCRIPTIONS = {
    "seq_diff":  "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "site_diff": "PPLM_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
}
_PREDICTOR_MAP = {"seq_diff": PPLMSeqDiff, "site_diff": PPLMSiteDiff}


def install_pplm_cache() -> None:
    print(f"Loading PPLM GCV cache: {_GCV_CACHE}", flush=True)
    with open(_GCV_CACHE, "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache)} entries", flush=True)
    cache_key = str(_GCV_CACHE)
    _pplm_mod.CACHE_PATH = cache_key
    PPLMSeqDiff._cache_path  = cache_key
    PPLMSiteDiff._cache_path = cache_key
    nn_base._CACHE_STORE[cache_key] = cache


def build_vc1pcava_test_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build PPLM-format DataFrame and canonical 0-based vt_ids from vc1pcava raw data.

    Returns (df, canonical_vt_ids) where df has 1-based mutations (for PPLM cache
    lookup) and canonical_vt_ids has 0-based mutations (for output npy files).
    """
    rows = []
    canonical_vt_ids = []

    for data, i, wt_id, variant, u1, u2, canon_id, label, c in iter_vc1pcava_entries(
        sources, sf_proteins
    ):
        pos_0 = int(variant[1:-1])
        mut_1based = f"{variant[0]}{pos_0 + 1}{variant[-1]}"  # for PPLM cache lookup

        rows.append({
            "interactor": u1,
            "partner":    u2,
            "mutation":   mut_1based,
            "perturbed":  label,
            "class":      c,
        })
        canonical_vt_ids.append(canon_id)

    print(f"  Unique entries: {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    return df, canonical_vt_ids


def run(predictor: str = "seq_diff", seed: int = 42) -> str:
    """Train/predict PPLM on VC1p+CAVA entries and save the vc1pcava supplement.

    Returns the method description string (also the key expected by
    merge_vc1pcava_into_main.merge_method() / restratify_vcfp_blind_test.restratify_one_method()).
    """
    t0 = time.time()
    description = _DESCRIPTIONS[predictor]

    install_pplm_cache()

    print("Building SF protein set…", flush=True)
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins", flush=True)

    sources = load_vc1pcava_sources()

    print("\nBuilding test DataFrame (UniProt remap + dedup)…", flush=True)
    test_df, canonical_vt_ids = build_vc1pcava_test_df(sources, sf_proteins)

    print("\nLoading SF training data…", flush=True)
    train_df = load_sf_train_df()
    print(f"  SF train: {len(train_df)} entries", flush=True)

    PredClass = _PREDICTOR_MAP[predictor]
    print(f"\nTraining {PredClass().name} on {len(train_df)} rows (seed={seed})…", flush=True)
    pred = PredClass(seed=seed)
    pred.fit(train_df)

    print(f"\nPredicting on {len(test_df)} vc1pcava entries…", flush=True)
    scores = pred.predict(test_df).astype(np.float32)

    result = save_vc1pcava_supplement(
        scores, test_df["perturbed"].values.astype(int), np.array(canonical_vt_ids),
        test_df["class"].values, description,
    )
    print(f"\nDone in {(time.time() - t0)/60:.1f} min", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="seq_diff", choices=("seq_diff", "site_diff"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(predictor=args.predictor, seed=args.seed)


if __name__ == "__main__":
    main()
