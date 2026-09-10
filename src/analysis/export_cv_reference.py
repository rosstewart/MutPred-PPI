#!/usr/bin/env python
"""Export the CV reference artifacts (row ordering, fold splits, test classes).

Several downstream analyses need the splits without re-running cross-validation,
and they are small enough to distribute (a few MB per dataset).

Driven entirely by the CANONICAL row tables. The previous version rebuilt the
ordering from `mutpred_ppi_cv.load_dataset`, which globbed `.mat` files and
carried a parallel `data` dict of embeddings, graphs and `vt_id` strings; that
loader is gone. The ordering is now just the canonical table's own row order, so
`row_index` means the same thing here, in the emitted rows file, and in the
tables every other consumer reads.

OUTPUT
    {prefix}rows.csv.gz            row_index, interactor, partner, mutation
    {prefix}fold_splits_{seed}.pkl [(fold, train_idx, test_idx)]
    {ptc_prefix}pair_test_classes_{seed}.npy   C1/C2/C3 per test row, fold order
    {prefix}clusters.pkl           cd-hit cluster id per row

`vt_id` pickles are NO LONGER written. They were
`'{interactor}-{partner} {mutation}'` composites with a 0-based mutation, and
that welding is ambiguous the moment an accession carries an isoform suffix --
261 of 2,785 ids in the sahni_fragoza reference contain more than one `-`.
Consumers join on explicit columns instead.

Usage:
    python src/analysis/export_cv_reference.py --dataset sahni_fragoza_mapped090826
    python src/analysis/export_cv_reference.py --dataset all
"""
from __future__ import annotations

import argparse
import csv
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np

from paths import DATASETS_DIR  # noqa: E402
from utils.gcv_common import (  # noqa: E402
    DATASET_CONFIGS, complex_clusters, compute_pair_test_classes, load_data,
    make_fold_splits,
)

OUT_DIR = DATASETS_DIR / "cv_reference"

# dataset -> (rows/fold_splits prefix, pair_test_classes prefix). The two differ
# because the class arrays were historically named after the SWING/combined run
# that first produced them; consumers still look them up by those names.
NAMING = {
    "sahni_fragoza_mapped090826":              ("sahni_fragoza_train_", "swing_train_"),
    "sahni_fragoza_varchamp_all_mapped090826": ("sahni_fragoza_varchamp_all_train_",
                                                "combined_sahni_fragoza_varchamp_all_"),
    "varchamp_all_mapped090826":               ("varchamp_all_train_",
                                                "combined_varchamp_all_"),
    "sahni_only_mapped090826":                 ("sahni_only_train_", "sahni_only_"),
    "fragoza_only_mapped090826":               ("fragoza_only_train_", "fragoza_only_"),
}


def _write_rows(path: Path, df) -> None:
    """The canonical ordering in explicit columns.

    `row_index` is the position in this ordering, so it indexes
    `fold_splits_{seed}.pkl` and `pair_test_classes_{seed}.npy` directly.
    """
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_index", "interactor", "partner", "mutation"])
        for idx, (i, p, m) in enumerate(zip(df["interactor"], df["partner"],
                                            df["mutation"])):
            w.writerow([idx, i, p, m])


def export(dataset: str, n_seeds: int = 30, identity: float = 0.5) -> None:
    prefix, ptc_prefix = NAMING[dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {dataset} ===", flush=True)
    df = load_data(DATASET_CONFIGS[dataset]).reset_index(drop=True)
    print(f"  {len(df)} rows", flush=True)

    # Groups are cd-hit clusters of the FULL COMPLEX sequence, never one chain.
    clusters = complex_clusters(df, identity=identity)
    print(f"  {len(set(clusters))} clusters at {identity:.0%} identity", flush=True)
    with open(OUT_DIR / f"{prefix}clusters.pkl", "wb") as f:
        pickle.dump(clusters, f)

    _write_rows(OUT_DIR / f"{prefix}rows.csv.gz", df)

    pairs = list(zip(df["interactor"], df["partner"]))
    for s in range(n_seeds):
        fold_splits = make_fold_splits(clusters, s)
        ptc = compute_pair_test_classes(pairs, fold_splits)
        with open(OUT_DIR / f"{prefix}fold_splits_{s}.pkl", "wb") as f:
            pickle.dump(fold_splits, f)
        np.save(OUT_DIR / f"{ptc_prefix}pair_test_classes_{s}.npy", ptc)
    counts = np.bincount(ptc, minlength=4)[1:]
    print(f"  wrote {n_seeds} seeds; last-seed class counts C1/C2/C3 = {counts}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sahni_fragoza_mapped090826",
                    choices=sorted(NAMING) + ["all"])
    ap.add_argument("--n-seeds", type=int, default=30)
    ap.add_argument("--identity", type=float, default=0.5,
                    help="cd-hit identity for the GroupKFold groups")
    args = ap.parse_args()

    targets = sorted(NAMING) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        export(ds, args.n_seeds, args.identity)
    print(f"\nOutput: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
