#!/usr/bin/env python
"""Export the canonical CV reference artifacts (orderings, fold splits, test classes).

These are generated inline by src/evaluation/mutpred_ppi_cv.py during a run, but
several downstream analyses need them without re-running cross-validation, and
they are small enough to distribute (a few MB per dataset). Writing them here
removes the last dependency on the external cv_splits directory and guarantees
consumers see the same class labels the GCV results were scored with.

File names match the historical cv_splits layout so existing consumers only need
their CV_DIR pointed at the output directory.

Usage:
    conda run -n ppi python src/analysis/export_cv_reference.py --dataset sahni_fragoza
    conda run -n ppi python src/analysis/export_cv_reference.py --dataset all
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "evaluation"))
from paths import DATASETS_DIR  # noqa: E402
import mutpred_ppi_cv as cv  # noqa: E402

OUT_DIR = DATASETS_DIR / "cv_reference"

# dataset -> (vt_ids/fold_splits prefix, pair_test_classes prefix)
NAMING = {
    "sahni":                              ("", ""),
    "sahni_fragoza":                      ("sahni_fragoza_train_", "swing_train_"),
    "sahni_fragoza_varchamp1p_cava":      ("sahni_fragoza_varchamp1p_cava_train_",
                                           "combined_sahni_fragoza_varchamp1p_cava_seq_confirmed_"),
    "sahni_varchamp1p_cava":              ("sahni_varchamp1p_cava_train_",
                                           "combined_sahni_varchamp1p_cava_seq_confirmed_concat_clust_"),
    "sahni_fragoza_varchamp_full":        ("sahni_fragoza_varchamp_full_train_",
                                           "combined_sahni_fragoza_varchamp_full_"),
    "sahni_fragoza_varchamp_pooled":      ("sahni_fragoza_varchamp_pooled_train_",
                                           "combined_sahni_fragoza_varchamp_pooled_"),
    "sahni_fragoza_varchamp_full_pooled": ("sahni_fragoza_varchamp_full_pooled_train_",
                                           "combined_sahni_fragoza_varchamp_full_pooled_"),
}


def export(dataset: str, n_seeds: int = 30) -> None:
    prefix, ptc_prefix = NAMING[dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset} ===", flush=True)
    cfg = cv.DATASET_CONFIGS[dataset]
    data = cv.load_dataset(cfg)
    canonical = cv.canonical_vt_ids_path(cfg)
    if canonical.exists():
        # Preserve the established ordering; never re-derive it, or fold splits
        # silently stop matching every previously published run.
        ordered = cv.align_to_vt_ids(data, canonical)
    else:
        print(f"  no canonical ordering yet; establishing one by shuffle", flush=True)
        ordered = cv.shuffle_data(data)
    vt_ids = ordered["all_vt_ids"]
    print(f"  {len(vt_ids)} rows", flush=True)

    with open(OUT_DIR / f"{prefix}all_vt_ids.pkl", "wb") as f:
        pickle.dump(vt_ids, f)
    for s in range(n_seeds):
        with open(OUT_DIR / f"{prefix}all_vt_ids_{s}.pkl", "wb") as f:
            pickle.dump(vt_ids, f)

    for s in range(n_seeds):
        fold_splits = cv.make_fold_splits(ordered, s)
        ptc = cv.compute_pair_test_classes(ordered, fold_splits)
        with open(OUT_DIR / f"{prefix}fold_splits_{s}.pkl", "wb") as f:
            pickle.dump(fold_splits, f)
        np.save(OUT_DIR / f"{ptc_prefix}pair_test_classes_{s}.npy", ptc)
    counts = np.bincount(ptc, minlength=4)[1:]
    print(f"  wrote {n_seeds} seeds; last-seed class counts C1/C2/C3 = {counts}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sahni_fragoza", choices=sorted(NAMING) + ["all"])
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()

    targets = sorted(NAMING) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        try:
            export(ds, args.n_seeds)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    print(f"\nOutput: {OUT_DIR}")
