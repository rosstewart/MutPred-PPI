#!/usr/bin/env python3
"""Restratify ALL VCFP blind test arrays using a single canonical sf_proteins.

Different blind test scripts were run at different times with slightly different
sf_exclusion logic, producing inconsistent C1/C2/C3 class assignments for the
same test pairs.  This script enforces a single standard:

  sf_proteins = union of interactor and partner UniProt IDs from
                training_data.csv rows where dataset contains "Sahni" or "Fragoza"

All methods' per-class npy files are re-split using this canonical sf_proteins.
vt_ids must be in "{interactor} {partner} {mutation}" format (3 space-sep fields).

Usage:
    conda run -n ppi python src/analysis/restratify_vcfp_blind_test.py [--dry-run]
"""

import argparse
import os
import numpy as np
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_CACHES_DIR, REPO_ROOT  # noqa: E402


_PUB = str(REPO_ROOT)
_EVAL   = os.path.join(_PUB, "results/varchamp_seqcnf_newvar_eval")
_CSV = str(DATA_CACHES_DIR / "training_data_internal.csv")
# All method descriptions that have VCFP blind test arrays
METHODS = [
    "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)",
    "MutPred-PPI (sahni, megascale_all, all-data) (varchamp_full_pooled)",
    "eSIG-Net (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SWING (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SWING (test pretrain, Sahni+Fragoza train) (varchamp_full_pooled)",
    "MutPPI (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MutPPIPlus (Sahni+Fragoza train) (varchamp_full_pooled)",
    "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "PPLM_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MINT_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MINT_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SAAMBE-3D (Sahni+Fragoza train) (varchamp_full_pooled)",
    "DDMutPPI (varchamp_full_pooled)",
]

# SKEMPI-trained methods use SAAMBE_train_uniprots instead of sf_proteins
_SAAMBE_UNIPROTS_F = os.path.join(_PUB, "results_revisions/macro_aucs/SAAMBE_train_uniprots.npy")
SKEMPI_METHODS = {
    "SAAMBE-3D (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MutPPI (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MutPPIPlus (Sahni+Fragoza train) (varchamp_full_pooled)",
    "DDMutPPI (varchamp_full_pooled)",
}


def build_sf_proteins() -> set:
    df = pd.read_csv(_CSV)
    sf_mask = df["dataset"].str.contains("Sahni") | df["dataset"].str.contains("Fragoza")
    return set(df.loc[sf_mask, "interactor"]) | set(df.loc[sf_mask, "partner"])


def classify(interactor: str, partner: str, train_proteins: set) -> int:
    int_in = interactor in train_proteins
    par_in = partner    in train_proteins
    if int_in and par_in:
        return 1
    if int_in or par_in:
        return 2
    return 3


def restratify(method: str, train_proteins: set, ref_vt_ids: set,
               dry_run: bool) -> None:
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Re-stratifying: {method}")

    all_preds, all_labels, all_vt_ids = [], [], []
    for c in [1, 2, 3]:
        pf = os.path.join(_EVAL, f"{method}_c{c}_preds.npy")
        lf = os.path.join(_EVAL, f"{method}_c{c}_labels.npy")
        vf = os.path.join(_EVAL, f"{method}_c{c}_vt_ids.npy")
        if not (os.path.exists(pf) and os.path.exists(lf) and os.path.exists(vf)):
            continue
        p = np.load(pf)
        if len(p) == 0:
            continue
        all_preds.append(p)
        all_labels.append(np.load(lf))
        all_vt_ids.extend(np.load(vf, allow_pickle=True))

    if not all_preds:
        print("  No data found — skipping")
        return

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    vt_ids = np.array(all_vt_ids)
    n_total = len(preds)
    print(f"  Combined: {n_total} pairs")

    # Reclassify using canonical training proteins
    new_classes = np.full(n_total, 3, dtype=int)
    bad_fmt = 0
    for i, vt in enumerate(vt_ids):
        parts = vt.split()
        if len(parts) < 3:
            bad_fmt += 1
            continue
        interactor, partner = parts[0], parts[1]
        new_classes[i] = classify(interactor, partner, train_proteins)

    if bad_fmt:
        print(f"  WARNING: {bad_fmt} vt_ids with unexpected format (treated as C3)")

    # Intersect with reference set to enforce consistent test-set membership
    # across methods that use different protein ID formats (e.g. SWING vs MutPred-PPI)
    if ref_vt_ids:
        in_ref = np.array([vt in ref_vt_ids for vt in vt_ids])
        n_dropped = int((~in_ref).sum())
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} entries not in reference test set")
        preds      = preds[in_ref]
        labels     = labels[in_ref]
        vt_ids     = vt_ids[in_ref]
        new_classes = new_classes[in_ref]

    for c in [1, 2, 3]:
        mask  = new_classes == c
        n_pos = int((labels[mask] == 1).sum())
        n_neg = int((labels[mask] == 0).sum())
        print(f"  New C{c}: n={mask.sum()} (pos={n_pos}, neg={n_neg})")

    if dry_run:
        return

    for c in [1, 2, 3]:
        mask = new_classes == c
        np.save(os.path.join(_EVAL, f"{method}_c{c}_preds.npy"),  preds[mask].astype(np.float32))
        np.save(os.path.join(_EVAL, f"{method}_c{c}_labels.npy"), labels[mask])
        np.save(os.path.join(_EVAL, f"{method}_c{c}_vt_ids.npy"), vt_ids[mask])

    print(f"  Saved.")


def build_ref_vt_ids(ref_method: str) -> set:
    """Build the canonical test-set vt_ids from the reference method (MutPred-PPI)."""
    ref_vts = set()
    for c in [1, 2, 3]:
        vf = os.path.join(_EVAL, f"{ref_method}_c{c}_vt_ids.npy")
        if os.path.exists(vf):
            ref_vts.update(np.load(vf, allow_pickle=True).tolist())
    return ref_vts


def restratify_one_method(method: str, dry_run: bool = False) -> None:
    """Restratify a single method's VCFP arrays using the canonical sf_proteins.

    Builds the same context main() builds (sf_proteins, SKEMPI train proteins,
    the MutPred-PPI reference vt_id set), then re-splits just `method`'s
    per-class npy files. Importable so callers (e.g. run_vcfp_blind_test.py)
    can restratify one method's arrays without invoking every method in
    METHODS. `method` need not be a member of METHODS — it uses the same
    SKEMPI_METHODS / _REF membership checks as main() to decide which
    canonical protein set and reference test-set intersection to apply.
    """
    sf_proteins     = build_sf_proteins()
    skempi_proteins = set(np.load(_SAAMBE_UNIPROTS_F, allow_pickle=True).tolist())

    _REF = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"
    ref_vt_ids = build_ref_vt_ids(_REF)

    if method in SKEMPI_METHODS:
        train_proteins = skempi_proteins
        use_ref = set()   # SKEMPI methods have different test sets; don't filter
    else:
        train_proteins = sf_proteins
        use_ref = ref_vt_ids if method != _REF else set()

    restratify(method, train_proteins, use_ref, dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sf_proteins    = build_sf_proteins()
    skempi_proteins = set(np.load(_SAAMBE_UNIPROTS_F, allow_pickle=True).tolist())

    # Reference test set: MutPred-PPI vt_ids define the canonical VCFP test set
    _REF = "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)"
    ref_vt_ids = build_ref_vt_ids(_REF)
    print(f"SF proteins: {len(sf_proteins)}")
    print(f"SKEMPI proteins: {len(skempi_proteins)}")
    print(f"Reference vt_ids (MutPred-PPI VCFP): {len(ref_vt_ids)}")

    for method in METHODS:
        if method in SKEMPI_METHODS:
            train_proteins = skempi_proteins
            use_ref = set()   # SKEMPI methods have different test sets; don't filter
        else:
            train_proteins = sf_proteins
            use_ref = ref_vt_ids if method != _REF else set()
        restratify(method, train_proteins, use_ref, args.dry_run)


if __name__ == "__main__":
    main()
