#!/usr/bin/env python3
"""Merge vc1pcava supplement npy files into the main VCFP blind test arrays.

For each method that has vc1pcava supplement files (*_vc1pcava_c{1,2,3}_*.npy),
concatenates them with the existing *_c{1,2,3}_*.npy files and overwrites.

Methods without vc1pcava supplements (SAAMBE-3D, MutPPI, MutPPIPlus, DDMutPPI)
are skipped — their blind test results remain at the 14,116-entry VCFP set.

Run BEFORE restratify_vcfp_blind_test.py.

Usage:
    conda run -n ppi python src/analysis/merge_vc1pcava_into_main.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import VCFP_RESULTS_DIR  # noqa: E402


_EVAL = VCFP_RESULTS_DIR
# Methods that have vc1pcava supplements
METHODS_WITH_SUPP = [
    "MutPred-PPI (megascale_all, all-data) (varchamp_full_pooled)",
    "MutPred-PPI (sahni, megascale_all, all-data) (varchamp_full_pooled)",
    "eSIG-Net (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SWING (Sahni+Fragoza train) (varchamp_full_pooled)",
    "SWING (test pretrain, Sahni+Fragoza train) (varchamp_full_pooled)",
    "MINT_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MINT_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "PPLM_seq_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "PPLM_site_diff (Sahni+Fragoza train) (varchamp_full_pooled)",
    "MutPred2 (varchamp_full_pooled)",
]


def merge_method(method: str, dry_run: bool) -> None:
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Merging: {method}")
    any_supp = False

    for c in [1, 2, 3]:
        main_pf = _EVAL / f"{method}_c{c}_preds.npy"
        main_lf = _EVAL / f"{method}_c{c}_labels.npy"
        main_vf = _EVAL / f"{method}_c{c}_vt_ids.npy"
        supp_pf = _EVAL / f"{method}_vc1pcava_c{c}_preds.npy"
        supp_lf = _EVAL / f"{method}_vc1pcava_c{c}_labels.npy"
        supp_vf = _EVAL / f"{method}_vc1pcava_c{c}_vt_ids.npy"

        if not supp_pf.exists():
            print(f"  C{c}: no supplement found — skipping")
            continue

        any_supp = True
        supp_p = np.load(supp_pf)
        supp_l = np.load(supp_lf)
        supp_v = np.load(supp_vf, allow_pickle=True)

        if main_pf.exists() and len(np.load(main_pf)) > 0:
            main_p = np.load(main_pf)
            main_l = np.load(main_lf)
            main_v = np.load(main_vf, allow_pickle=True)

            # Skip supplement entries already in main (idempotent — safe to run multiple times)
            main_set = set(main_v.tolist())
            new_mask = np.array([v not in main_set for v in supp_v])
            n_dup = int((~new_mask).sum())
            if n_dup:
                print(f"  C{c}: skipping {n_dup} supplement entries already in main")
            supp_p = supp_p[new_mask]
            supp_l = supp_l[new_mask]
            supp_v = supp_v[new_mask]

            if len(supp_p) == 0:
                print(f"  C{c}: all supplement entries already present — nothing to add")
                continue

            merged_p = np.concatenate([main_p, supp_p]).astype(np.float32)
            merged_l = np.concatenate([main_l, supp_l])
            merged_v = np.concatenate([main_v, supp_v])
            print(f"  C{c}: main={len(main_p)} + supp={len(supp_p)} → {len(merged_p)}")
        else:
            merged_p = supp_p.astype(np.float32)
            merged_l = supp_l
            merged_v = supp_v
            print(f"  C{c}: no main file — supp only: {len(merged_p)}")

        if not dry_run:
            np.save(main_pf, merged_p)
            np.save(main_lf, merged_l)
            np.save(main_vf, merged_v)

    if not any_supp:
        print("  No supplement files found for any class — nothing merged")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = ap.parse_args()

    for method in METHODS_WITH_SUPP:
        merge_method(method, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
