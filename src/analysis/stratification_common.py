"""The GCV fold-curve reconstruction shared by every stratified supplement.

`protein_class_stratification.py`, `plddt_stratification.py` and
`interface_analysis.py` all answer the same question -- "how do the per-class
GCV ROC curves split when rows are grouped by X?" -- and differed only in what
X is. They each carried their own ~90-line copy of the reconstruction below,
plus a ~35-line copy of the per-group curve aggregation.

Consolidated 2026-09-10. The copies had already drifted: only
`protein_class_stratification` still asserted that `row_index` equals the
DataFrame position and that each seed's fold splits cover exactly the canonical
row count. The other two indexed positionally anyway, on the strength of a
comment. Both guards now apply everywhere, which is the point of having one
copy.

Every consumer uses the same canonical inputs (the Sahni+Fragoza GCV pkl, 30
seeds, the exported CV reference), so those are constants here rather than
parameters.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from analysis.gcv_curves import FPR_GRID
from paths import REPO_ROOT, cv_reference_dir
from utils.gcv_common import StaleCacheError, load_gcv_detailed_results

CV_DIR = str(cv_reference_dir())
GCV_RESULTS = (f"{REPO_ROOT}/results/gcv/"
               f"MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl")
CANONICAL_DATASET = "sahni_fragoza_mapped090826"
ROWS_FILE = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"

N_SEEDS = 30
MIN_N = 5          # per fold+group: enough rows, and both labels present
CLASSES = (1, 2, 3)


def load_canonical_rows(rows_file: str = ROWS_FILE) -> pd.DataFrame:
    """The canonical row table, with `row_index` verified to be 0..n-1.

    Everything downstream indexes the fold splits and the test-class array
    POSITIONALLY, so this equivalence is load-bearing rather than incidental.
    """
    rows = pd.read_csv(rows_file).sort_values("row_index").reset_index(drop=True)
    if list(rows["row_index"]) != list(range(len(rows))):
        raise ValueError(
            f"{rows_file}: row_index is not 0..n-1, so it cannot be used as a "
            f"positional index into the fold splits. Regenerate with "
            f"src/analysis/export_cv_reference.py.")
    return rows


def stratified_fold_curves(row_groups, groups, *, min_n: int = MIN_N,
                           n_seeds: int = N_SEEDS, rows_file: str = ROWS_FILE,
                           gcv_results: str = GCV_RESULTS,
                           canonical_dataset: str = CANONICAL_DATASET):
    """Per-(test class, group) interpolated ROC curves, one per GCV fold.

    `row_groups` is a per-row array of group labels, positionally aligned to the
    canonical rows (use `load_canonical_rows` to build it). `groups` is the
    ordered list of labels to report; rows whose label is not in it (typically
    `None`, meaning "unknown") are simply never selected.

    Returns `(fold_curves, all_rows_by_group)`:
      fold_curves[class][group]      -> list of TPR arrays on FPR_GRID
      all_rows_by_group[class][group] -> set of row_index values contributing
    """
    rows = load_canonical_rows(rows_file)
    row_groups = np.asarray(row_groups, dtype=object)
    if len(row_groups) != len(rows):
        raise ValueError(f"row_groups has {len(row_groups)} entries but "
                         f"{os.path.basename(rows_file)} has {len(rows)} rows")

    gcv = load_gcv_detailed_results(gcv_results, canonical_dataset)

    fold_curves = {c: {g: [] for g in groups} for c in CLASSES}
    all_rows_by_group = {c: {g: set() for g in groups} for c in CLASSES}

    for seed in range(n_seeds):
        fold_splits_path = f"{CV_DIR}/sahni_fragoza_train_fold_splits_{seed}.pkl"
        ptc_path = f"{CV_DIR}/swing_train_pair_test_classes_{seed}.npy"
        if not all(os.path.exists(p) for p in (fold_splits_path, ptc_path)):
            print(f"  Seed {seed}: missing CV reference files, skipping", flush=True)
            continue

        with open(fold_splits_path, "rb") as f:
            fold_splits = pickle.load(f)
        pair_test_classes = np.load(ptc_path)

        n_test_total = sum(len(t) for _, _, t in fold_splits)
        if n_test_total != len(rows) or len(pair_test_classes) != len(rows):
            raise ValueError(
                f"seed {seed}: fold splits cover {n_test_total} rows and the "
                f"test-class array {len(pair_test_classes)}, but "
                f"{os.path.basename(rows_file)} has {len(rows)}. Regenerate the "
                f"CV reference with src/analysis/export_cv_reference.py "
                f"--dataset {canonical_dataset}.")

        iteration = gcv["iterations"][seed]
        flat_cursor = 0

        for fold, _train_idx, test_idx in sorted(fold_splits, key=lambda t: t[0]):
            fold_data = iteration["folds"][fold]
            n_test = len(test_idx)
            ptc_fold = pair_test_classes[flat_cursor:flat_cursor + n_test]
            flat_cursor += n_test

            preds_fold = {c: list(fold_data[f"class_{c}"]["preds"]) for c in CLASSES}
            labels_fold = {c: list(fold_data[f"class_{c}"]["labels"]) for c in CLASSES}

            # Freshness of the pkl AS A WHOLE is asserted once by
            # `load_gcv_detailed_results`, comparing one seed's TOTAL row count
            # against the canonical table. A global total can match by
            # construction while an individual fold's class buckets still
            # disagree, and the interleave below indexes those buckets
            # unconditionally -- so a per-fold mismatch must be named here or it
            # surfaces as an opaque IndexError.
            n_cached = sum(len(v) for v in preds_fold.values())
            if n_cached != n_test:
                raise StaleCacheError(
                    f"{os.path.basename(gcv_results)}: seed {seed} fold {fold} "
                    f"holds {n_cached} cached predictions but the canonical fold "
                    f"has {n_test} test rows, despite the pkl's overall row count "
                    f"matching. This fold is internally inconsistent and must be "
                    f"recomputed.")

            cursor = {c: 0 for c in CLASSES}
            preds_ordered, labels_ordered = [], []
            for cls in ptc_fold:
                preds_ordered.append(preds_fold[cls][cursor[cls]])
                labels_ordered.append(labels_fold[cls][cursor[cls]])
                cursor[cls] += 1

            preds_ordered = np.array(preds_ordered)
            labels_ordered = np.array(labels_ordered)
            classes_ordered = np.array(ptc_fold)
            fold_rows = np.asarray(test_idx)
            groups_ordered = row_groups[fold_rows]

            for cls in CLASSES:
                mask_cls = classes_ordered == cls
                for grp in groups:
                    mask = mask_cls & (groups_ordered == grp)
                    p, l = preds_ordered[mask], labels_ordered[mask]
                    all_rows_by_group[cls][grp].update(fold_rows[mask].tolist())
                    if len(p) >= min_n and len(np.unique(l)) == 2:
                        fpr, tpr, _ = roc_curve(l, p)
                        fold_curves[cls][grp].append(np.interp(FPR_GRID, fpr, tpr))

    return fold_curves, all_rows_by_group


def mean_curve_and_auc(curves):
    """`(mean_tpr, sem_tpr, mean_auc, n_folds)` for one group's fold curves.

    The mean curve is forced through (0,0) and (1,1); AUC is the trapezoid of
    the mean curve, not the mean of per-fold AUCs, matching `gcv_curves`.
    """
    if not curves:
        return None, None, float("nan"), 0
    arr = np.vstack(curves)
    mean_tpr = arr.mean(axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    sem_tpr = arr.std(axis=0) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean_tpr)
    return mean_tpr, sem_tpr, auc(FPR_GRID, mean_tpr), len(curves)
