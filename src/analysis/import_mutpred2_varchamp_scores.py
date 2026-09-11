#!/usr/bin/env python
"""Import MutPred2 scores for the VarChAMP blind test.

MutPred2 has no trainable model to run in-repo -- it is an external tool
whose output is a CSV of (protein, 1-based mutation, score) scored offline.
This parses that CSV and joins it against the canonical
`varchamp_all_mapped090826` rows directly.

There is exactly one canonical test set (`varchamp_all_mapped090826`) and no
supplementing, so there is nothing to merge and nothing to restratify: this
script computes the join and the classing once, directly. Result files that
predate that table are rejected by `utils.legacy_guard`.

MutPred2 is partner-agnostic (one score per (protein, mutation), scored
without reference to a partner), so a single CSV row scores every
`varchamp_all_mapped090826` row that shares its (interactor, mutation) --
however many partners that variant was tested against.
`varchamp_blind_test.STRATIFICATION_INDEPENDENT_METHODS` pools its C1/C2/C3
arrays back together for exactly this reason; the class assigned here is a
real overlap classification (via the same `compute_blind_test_classes` the
other methods use), consistent with everything else, even though the figure
script itself does not use it to stratify MutPred2's ROC curve.

Usage:
    conda run -n ppi python src/analysis/import_mutpred2_varchamp_scores.py \\
        --csv mutpred2_varchamp_output.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent  # src/analysis
_EVAL_DIR = _ANALYSIS_DIR.parent / "evaluation"
sys.path.insert(0, str(_EVAL_DIR))

from run_varchamp_blind_test import TEST_CFG, load_train_test, save_results  # noqa: E402
from analysis.mutpred2_common import parse_mutpred2_csv as load_mutpred2_csv  # noqa: E402
from utils.gcv_common import compute_blind_test_classes  # noqa: E402

_MP2_METHOD = "MutPred2 (varchamp_blind_test)"


def run(csv_path: Path, dry_run: bool = False) -> None:
    print(f"Loading MutPred2 output: {csv_path}")
    mp2_scores = load_mutpred2_csv(csv_path)
    print(f"  {len(mp2_scores)} (protein, mutation) scores loaded")

    train_df, test_df = load_train_test()
    print(f"  test set ({TEST_CFG.name}): {len(test_df)} rows")

    scores = np.array(
        [mp2_scores.get((row.interactor, row.mutation), np.nan)
         for row in test_df.itertuples()],
        dtype=float)
    n_found = int(np.sum(~np.isnan(scores)))
    print(f"  matched: {n_found}/{len(test_df)} rows "
          f"({len(test_df) - n_found} missing from the MutPred2 CSV)")

    if dry_run:
        print("\n[dry-run] Not writing files")
        return

    classes = compute_blind_test_classes(
        list(zip(train_df["interactor"], train_df["partner"])),
        list(zip(test_df["interactor"], test_df["partner"])))
    save_results(scores, test_df, classes, _MP2_METHOD)
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="MutPred2 output CSV file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse/report only -- skip writing files.")
    args = ap.parse_args()
    run(Path(args.csv), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
