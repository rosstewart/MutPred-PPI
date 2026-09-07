#!/usr/bin/env python
"""eSIG-Net supplemental blind test on VC1p + CAVA entries.

Proteins are remapped to UniProt via gene_symbol_to_uniprot.pkl. The ESM-2
cache is extended with precomputed supplement PKLs that cover vc1pcava mutant
and WT embeddings not already in esm2_residue_embeddings_with_pooled.pkl.

22 entries shared between VC1p and CAVA are deduplicated (by UniProt vt_id).

`run()` is importable (e.g. from src/evaluation/run_vcfp_blind_test.py) and
returns the method description string used for the saved npy files.

Usage:
    conda run -n ppi python supplement_esignet_vc1pcava.py [--device cuda:1] [--seed 42]
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

from vcfp_common import (  # noqa: E402
    TRAINING_CSV, build_sf_proteins, load_vc1pcava_sources,
    iter_vc1pcava_entries, save_vc1pcava_supplement,
)
import predictors.esignet as _esignet_mod          # noqa: E402
from predictors.esignet import ESigNetPredictor    # noqa: E402

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import REVISIONS_DIR  # noqa: E402


_ESM_CACHE = str(REVISIONS_DIR / "esm2_residue_embeddings_with_pooled.pkl")
_VC1P_SUPP = Path(
    "/data/ross/ppi_lossgain/interaction_loss/home/eSIG-Net/"
    "esm2_varchamp1p_blind_test_supplement_sahni_fragoza.pkl"
)
_CAVA_SUPP = Path(
    "/data/ross/ppi_lossgain/interaction_loss/home/eSIG-Net/"
    "esm2_cava_blind_test_supplement_sahni_fragoza.pkl"
)
_DESCRIPTION = "eSIG-Net (Sahni+Fragoza train) (varchamp_full_pooled)"


def build_test_df(
    sources: list[tuple[dict, dict]],
    sf_proteins: set,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Build test DataFrame (UniProt + 1-based mutations for eSIG-Net lookup).

    Returns:
        df: rows with interactor/partner/mutation/sequences/perturbed
        uniprot_vt_ids: "UNIPROT1 UNIPROT2 MUT_0based" (canonical format for npy)
        class_labels: C1/C2/C3 integers
    """
    rows, uniprot_vt_ids, class_labels = [], [], []

    for data, i, wt_id, variant_0based, u1, u2, canonical_id, label, c in iter_vc1pcava_entries(
        sources, sf_proteins
    ):
        # eSIG-Net expects 1-based mutation positions in the DataFrame
        pos_0 = int(variant_0based[1:-1])
        mut_1based = f"{variant_0based[0]}{pos_0 + 1}{variant_0based[-1]}"

        full_seq = data["wt_seqs"][i]
        L_inter  = data["seq_lengths"][i][0]

        rows.append({
            "interactor":          u1,
            "partner":             u2,
            "mutation":            mut_1based,
            "interactor_sequence": full_seq[:L_inter],
            "partner_sequence":    full_seq[L_inter:],
            "perturbed":           label,
        })
        uniprot_vt_ids.append(canonical_id)   # 0-based for output vt_id
        class_labels.append(c)

    print(f"  Unique entries: {len(rows)}")
    return pd.DataFrame(rows), uniprot_vt_ids, class_labels


def run(device: str = "", seed: int = 42) -> str:
    """Train/predict eSIG-Net on VC1p+CAVA entries and save the vc1pcava supplement.

    Returns the method description string (also the key expected by
    merge_vc1pcava_into_main.merge_method() / restratify_vcfp_blind_test.restratify_one_method()).
    """
    t0 = time.time()

    print("Building SF protein set…")
    sf_proteins = build_sf_proteins()
    print(f"  {len(sf_proteins)} SF UniProt proteins")

    sources = load_vc1pcava_sources()

    print("\nBuilding test DataFrame (UniProt remap + dedup)…")
    test_df, uniprot_vt_ids, class_labels = build_test_df(sources, sf_proteins)

    print("\nLoading main ESM-2 cache…")
    main_cache = _esignet_mod.load_cache(_ESM_CACHE)
    n_main = len(main_cache) if main_cache else 0
    print(f"  Main cache: {n_main} entries")

    print("Loading vc1p+cava supplement caches…")
    vc1p_supp = pickle.load(open(_VC1P_SUPP, "rb"))
    cava_supp = pickle.load(open(_CAVA_SUPP, "rb"))
    print(f"  VC1p supplement: {len(vc1p_supp)} entries")
    print(f"  CAVA supplement: {len(cava_supp)} entries")

    # Merge: supplement fills in vc1pcava entries missing from main cache
    merged_cache: dict = {} if main_cache is None else dict(main_cache)
    merged_cache.update(vc1p_supp)
    merged_cache.update(cava_supp)
    print(f"  Merged cache: {len(merged_cache)} entries")

    # Monkeypatch load_cache to return merged dict for any path
    _esignet_mod.load_cache = lambda path: merged_cache
    _esignet_mod._ESM_CACHE_PATH = _ESM_CACHE
    ESigNetPredictor.cache_paths = [_ESM_CACHE]

    print("\nLoading SF training data…")
    df_train = pd.read_csv(TRAINING_CSV)
    sf_mask = df_train["dataset"].str.contains("Sahni") | df_train["dataset"].str.contains("Fragoza")
    train_df = df_train[sf_mask].copy().reset_index(drop=True)
    # Drop NaN sequences
    train_df = train_df[
        train_df["interactor_sequence"].notna() & train_df["partner_sequence"].notna()
    ].reset_index(drop=True)
    print(f"  SF train: {len(train_df)} entries")

    print(f"\nTraining eSIG-Net (seed={seed})…")
    predictor = ESigNetPredictor(seed=seed, device=device)
    predictor.fit(train_df)

    print(f"\nPredicting on {len(test_df)} vc1pcava entries…")
    scores = predictor.predict(test_df).astype(np.float32)

    n_nan = int(np.isnan(scores).sum())
    if n_nan:
        print(f"  WARNING: {n_nan}/{len(scores)} NaN scores (missing cache entries)")

    # Drop NaN scores
    valid_mask = ~np.isnan(scores)
    scores_v       = scores[valid_mask]
    labels_v       = np.array([r["perturbed"] for r in test_df.to_dict("records")], dtype=np.int32)[valid_mask]
    vt_ids_v       = np.array(uniprot_vt_ids)[valid_mask]
    classes_v      = np.array(class_labels, dtype=np.int32)[valid_mask]

    print(f"\nTotal valid: {len(scores_v)} / {len(scores)}")
    result = save_vc1pcava_supplement(scores_v, labels_v, vt_ids_v, classes_v, _DESCRIPTION)

    print(f"\nDone in {(time.time() - t0)/60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    run(device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
