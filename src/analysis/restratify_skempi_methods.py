#!/usr/bin/env python3
"""Re-stratify SAAMBE-3D / MutPPI / MutPPI+ blind-test arrays using SKEMPI training proteins.

These methods are trained on SKEMPI, not Sahni+Fragoza. Their C1/C2/C3 classes
should reflect how well they generalise from THEIR own training set, i.e.
C1 = both proteins in SKEMPI training, C2 = one in, C3 = neither in.

The canonical SKEMPI reference is SAAMBE_train_uniprots.npy (258 proteins),
already used for the GCV-figure stratification in roc_plots.py.

**Redundant for freshly-generated arrays (2026-09-10).**
`run_varchamp_blind_test.py` now assigns SKEMPI-based C1/C2/C3 at generation
time itself (`utils.gcv_common.skempi_test_class`), the same rule this script
applies after the fact -- so re-running the blind test never needs this
script any more. It remains useful for arrays saved before that fix, or a
one-off consistency check; running it on fresh output is a harmless no-op
(the classes it computes will already match).

Usage:
    conda run -n ppi python src/analysis/restratify_skempi_methods.py [--dry-run]
"""

import argparse
import os
import numpy as np

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import GCV_RESULTS_DIR, REPO_ROOT, VARCHAMP_BLIND_TEST_DIR  # noqa: E402
from utils.blind_test_common import load_class_arrays  # noqa: E402


_PUB = str(REPO_ROOT)
_EVAL_DIR = str(VARCHAMP_BLIND_TEST_DIR)
_SAAMBE_UNIPROTS = str(GCV_RESULTS_DIR / "SAAMBE_train_uniprots.npy")

METHODS = [
    "SAAMBE-3D (Sahni+Fragoza train) (varchamp_blind_test)",
    "MutPPI (Sahni+Fragoza train) (varchamp_blind_test)",
    "MutPPIPlus (Sahni+Fragoza train) (varchamp_blind_test)",
]


def restratify(method: str, skempi_proteins: set, dry_run: bool) -> None:
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Re-stratifying: {method}")

    # Load existing per-class arrays and combine
    all_preds, all_labels, all_vt_ids = [], [], []
    for c in [1, 2, 3]:
        loaded = load_class_arrays(method, c, _EVAL_DIR, require_vt_ids=True)
        if loaded is None:
            print(f"  WARNING: missing c{c} files — skipping")
            continue
        p, l, v = loaded
        if len(p) == 0:
            print(f"  c{c}: empty — skipping")
            continue
        all_preds.append(p)
        all_labels.append(l)
        all_vt_ids.extend(v)

    if not all_preds:
        print("  No data found — skipping method")
        return

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    vt_ids = np.array(all_vt_ids)
    n_total = len(preds)
    print(f"  Combined: {n_total} pairs")

    # Re-classify C1/2/3 using SKEMPI training proteins
    # vt_ids format: "{interactor} {partner} {mutation}" (3 fields, space-separated)
    new_classes = np.full(n_total, 3, dtype=int)  # default C3
    for i, vt in enumerate(vt_ids):
        parts = vt.split()
        if len(parts) < 3:
            print(f"  WARNING: unexpected vt_id format '{vt}' — treating as C3")
            continue
        interactor = parts[0]
        partner    = parts[1]
        int_in = interactor in skempi_proteins
        par_in = partner    in skempi_proteins
        if int_in and par_in:
            new_classes[i] = 1
        elif int_in or par_in:
            new_classes[i] = 2
        else:
            new_classes[i] = 3

    # Report counts
    old_classes_approx = "previous (sf-based)"
    print(f"  Old ({old_classes_approx}): C1≈{sum(len(p) for p in all_preds if False)}")
    for c in [1, 2, 3]:
        n = (new_classes == c).sum()
        n_pos = int((labels[new_classes == c] == 1).sum())
        n_neg = int((labels[new_classes == c] == 0).sum())
        print(f"  New C{c}: n={n} (pos={n_pos}, neg={n_neg})")

    if dry_run:
        print("  [DRY-RUN] Would overwrite files — skipping writes")
        return

    # Save corrected stratified arrays (overwrite in-place)
    for c in [1, 2, 3]:
        mask = new_classes == c
        np.save(os.path.join(_EVAL_DIR, f"{method}_c{c}_preds.npy"),
                preds[mask].astype(np.float32))
        np.save(os.path.join(_EVAL_DIR, f"{method}_c{c}_labels.npy"),
                labels[mask])
        np.save(os.path.join(_EVAL_DIR, f"{method}_c{c}_vt_ids.npy"),
                vt_ids[mask])
    print(f"  Saved — {n_total} pairs re-stratified into new C1/C2/C3")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Print counts without writing files")
    args = p.parse_args()

    skempi_proteins = set(
        np.load(_SAAMBE_UNIPROTS, allow_pickle=True).tolist()
    )
    print(f"SKEMPI training proteins: {len(skempi_proteins)}")

    for method in METHODS:
        restratify(method, skempi_proteins, args.dry_run)


if __name__ == "__main__":
    main()
