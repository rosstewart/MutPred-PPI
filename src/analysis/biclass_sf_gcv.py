#!/usr/bin/env python3
"""Biclass SF GCV: filter Sahni+Fragoza GCV to ordered protein pairs where mutations
in the interactor span BOTH label=1 (disrupting) AND label=0 (non-disrupting).

A biclass pair (A, B) is ordered: mutations in protein A must have both classes; the
reverse pair (B, A) is evaluated independently. This captures pairs where the interaction
can be either maintained or disrupted, depending on which specific residue is mutated.

Output:
    results/biclass_gcv/roc_sahni_fragoza_biclass_with_variance.png

Usage:
    conda run -n ppi python src/analysis/biclass_sf_gcv.py
"""
from __future__ import annotations

import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import CV_DIR, REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
_CV = CV_DIR
from roc_plots import (compute_roc_with_variance, plot_roc_with_confidence,
                       METHOD_DISPLAY_NAMES, WORKING_DIR)
from utils.gcv_common import (
    DATASET_CONFIGS, StaleCacheError, load_data, load_gcv_detailed_results,
    load_positional_cache)


CANONICAL_DATASET = "sahni_fragoza_mapped090826"
PKL_DIR      = _PUB / "results" / "gcv"
OUT_DIR      = _PUB / "results" / "biclass_gcv"

DATASET = "sahni_fragoza"
N_SEEDS = 30

# Fixed-prediction baselines (not GCV-iterated) — evaluated by row position, so
# their arrays must be computed under the canonical `row_index` order that
# `load_pairs_in_row_order` reads.
SKEMPI_METHODS = ["SAAMBE-3D", "MutPPI", "MutPPIPlus"]  # DDMutPPI excluded entirely: 87% job-timeout rate, see docs/METHOD_PROVENANCE.md

# roc_plots.py's main() only registers these display names at runtime (line ~1083),
# so biclass_sf_gcv.py must register them itself before calling plot_roc_with_confidence.
METHOD_DISPLAY_NAMES.update({
    f"mutpred2_standalone_{DATASET}": "MutPred2",
    f"saambe_3d_{DATASET}":           "SAAMBE-3D",
    f"mutppi_{DATASET}":              "MutPPI",
    f"mutppiplus_{DATASET}":          "MutPPI+",
})

METHODS = [
    ("MutPredPPI_sahni_fragoza_megascale_all",   "MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"),
    ("ESigNet_sahni_fragoza",                    "ESigNet_sahni_fragoza_detailed_results.pkl"),
    ("SWING_sahni_fragoza_test_pretrain",        "SWING_sahni_fragoza_test_pretrain_detailed_results.pkl"),
    ("SWING_sahni_fragoza_no_test_pretrain",     "SWING_sahni_fragoza_no_test_pretrain_detailed_results.pkl"),
    ("MINT_seq_diff_sahni_fragoza",              "MINT_seq_diff_sahni_fragoza_detailed_results.pkl"),
    ("MINT_site_diff_sahni_fragoza",             "MINT_site_diff_sahni_fragoza_detailed_results.pkl"),
    ("PPLM_seq_diff_sahni_fragoza",              "PPLM_seq_diff_sahni_fragoza_detailed_results.pkl"),
    ("PPLM_site_diff_sahni_fragoza",             "PPLM_site_diff_sahni_fragoza_detailed_results.pkl"),
]


def load_biclass_pairs() -> set[tuple[str, str]]:
    """Ordered `(interactor, partner)` pairs whose mutations span labels 0 and 1.

    Derived from the canonical row table's `perturbed` column. This previously
    read `sahni_fragoza_all_vt_ids_and_labels.txt`, a pre-090826 artifact with no
    producer in the tree and no copy in git -- 3,014 of its 5,894 rows name pairs
    the 090826 remapping dropped outright, so it describes a row set that no
    longer exists. On the 2,874 rows the two share the labels agree exactly
    (0 disagreements), so nothing is lost by sourcing them here instead.
    """
    df = load_data(DATASET_CONFIGS[CANONICAL_DATASET])
    pair_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    for interactor, partner, label in zip(df["interactor"], df["partner"],
                                          df["perturbed"]):
        pair_labels[(str(interactor), str(partner))].add(int(label))
    biclass = {pair for pair, lbls in pair_labels.items() if lbls == {0, 1}}
    n_total = len(pair_labels)
    print(f"  Canonical rows:      {len(df):,}")
    print(f"  Total ordered pairs: {n_total:,}")
    print(f"  Biclass pairs:       {len(biclass):,} ({len(biclass)/n_total:.1%})")
    return biclass


def build_row_ids_by_seed(canonical_rows: pd.DataFrame, n_seeds: int
                          ) -> dict[int, dict[int, dict[str, list[tuple[str, str]]]]]:
    """`{seed: {fold: {class_k: [(interactor, partner), ...]}}}`, in the same
    per-fold/class order every GCV method pkl stores its `preds`/`labels`.

    Replaces a byproduct pickle (`iptm_{dataset}_gcv_splits.pkl`) that used to
    supply this same `(interactor, partner)` bookkeeping: it existed only to
    carry ipTM/pTM scores this script never used (that analysis, `iptm_analysis.py`,
    is not referenced by any manuscript figure), and its own row/fold structure
    had gone stale relative to the canonical table -- the byproduct inherited a
    staleness problem that has nothing to do with what this script needs. The
    canonical row table plus the fold splits and per-row test classes already
    used by `interface_analysis.py` / `plddt_stratification.py` /
    `export_reconstruction_tables.py` are the single source of truth for which
    pair sits at which (seed, fold, class) position.
    """
    result: dict[int, dict[int, dict[str, list[tuple[str, str]]]]] = {}
    for seed in range(n_seeds):
        fold_splits_path = Path(_CV) / f"sahni_fragoza_train_fold_splits_{seed}.pkl"
        ptc_path = Path(_CV) / f"swing_train_pair_test_classes_{seed}.npy"
        if not (fold_splits_path.exists() and ptc_path.exists()):
            continue
        with open(fold_splits_path, "rb") as f:
            fold_splits = pickle.load(f)
        ptc = np.load(ptc_path)

        seed_result: dict[int, dict[str, list[tuple[str, str]]]] = {}
        flat_cursor = 0
        for fold, _train_idx, test_idx in sorted(fold_splits, key=lambda t: t[0]):
            n_test = len(test_idx)
            ptc_fold = ptc[flat_cursor:flat_cursor + n_test]
            per_class: dict[int, list[tuple[str, str]]] = {1: [], 2: [], 3: []}
            for ridx, cls in zip(test_idx, ptc_fold):
                row = canonical_rows.iloc[ridx]
                per_class[int(cls)].append((str(row["interactor"]), str(row["partner"])))
            seed_result[fold] = {f"class_{c}": per_class[c] for c in (1, 2, 3)}
            flat_cursor += n_test
        result[seed] = seed_result
    return result


def filter_detailed_results(
    detailed_results: dict,
    row_ids_by_seed: dict[int, dict[int, dict[str, list[tuple[str, str]]]]],
    biclass_pairs: set[tuple[str, str]],
) -> dict:
    """Build a new detailed_results dict restricted to biclass-pair entries."""
    filtered: dict = {"iterations": {}}
    n_kept = n_total = 0

    for it, iter_data in detailed_results["iterations"].items():
        filtered["iterations"][it] = {"folds": {}}
        row_ids_iter = row_ids_by_seed.get(it, {})

        for fd, fold_data in iter_data["folds"].items():
            filtered["iterations"][it]["folds"][fd] = {}
            row_ids_fold = row_ids_iter.get(fd, {})

            for cl in ["class_1", "class_2", "class_3"]:
                orig = fold_data[cl]
                cids = row_ids_fold.get(cl, [])

                n_expected = len(cids)
                n_pred = len(orig["preds"])

                if n_expected != n_pred:
                    # Length mismatch: this method's fold does not align with the
                    # canonical fold structure — skip (empty entry, no AUC contribution)
                    filtered["iterations"][it]["folds"][fd][cl] = {
                        "preds": np.array([]), "labels": np.array([]), "auc": 0.0
                    }
                    continue

                mask   = np.array([c in biclass_pairs for c in cids])
                preds  = np.array(orig["preds"])[mask]
                labels = np.array(orig["labels"])[mask]

                n_kept  += int(mask.sum())
                n_total += len(mask)

                filtered["iterations"][it]["folds"][fd][cl] = {
                    "preds": preds, "labels": labels, "auc": 0.0
                }

    frac = n_kept / n_total if n_total else 0
    print(f"  Entries kept: {n_kept:,}/{n_total:,} ({frac:.1%})", flush=True)
    return filtered


def load_pairs_in_row_order() -> list[tuple[str, str]]:
    """`(interactor, partner)` per canonical row, in canonical row order.

    The fixed-prediction baseline arrays (SAAMBE-3D, MutPPI, MutPPIPlus,
    mutpred2_standalone) are keyed by ROW POSITION, so they are only meaningful
    against the ordering they were computed under. That used to be the raw line
    order of `sahni_fragoza_all_vt_ids_and_labels.txt`; it is now the `row_index`
    order of `sahni_fragoza_train_rows.csv.gz`, which `export_cv_reference.py`
    writes and every other consumer already joins on.

    The two orderings differ (5,894 vs 6,219 rows), so the existing `.npy`
    caches cannot be reindexed onto this one -- they must be recomputed. The
    caller raises rather than skipping so that never happens silently.
    """
    rows = pd.read_csv(_CV / "sahni_fragoza_train_rows.csv.gz")
    rows = rows.sort_values("row_index")
    return list(zip(rows["interactor"].astype(str), rows["partner"].astype(str)))


def filter_baseline_predictions(dataset: str, biclass_pairs: set[tuple[str, str]],
                                 complex_ids: list[tuple[str, str]]) -> dict:
    """Mirror roc_plots.load_baseline_predictions(), restricted to biclass pairs.

    Positional-cache staleness is checked by the one shared implementation,
    `utils.gcv_common.load_positional_cache` -- each array is checked
    independently against the canonical row count, which is equivalent to (and
    pinpoints the culprit better than) the previous three-way equality check.
    """
    baseline_results: dict = {}
    biclass_mask_all = np.array([c in biclass_pairs for c in complex_ids])
    n_rows = len(complex_ids)

    labels_file       = os.path.join(WORKING_DIR, f"{dataset}_mutpred2_standalone_labels.npy")
    test_classes_file = os.path.join(WORKING_DIR, f"{dataset}_SAAMBE-3D_test_classes.npy")
    labels_all = load_positional_cache(labels_file, n_rows)
    test_classes_all = load_positional_cache(test_classes_file, n_rows)
    if labels_all is None or test_classes_all is None:
        print("  [SKIP] baseline labels/test_classes files not found", flush=True)
        return baseline_results

    for method in SKEMPI_METHODS:
        preds_file = os.path.join(WORKING_DIR, f"{dataset}_{method}_preds.npy")
        preds_all = load_positional_cache(preds_file, n_rows)
        if preds_all is None:
            print(f"  [SKIP] {method}: preds file not found", flush=True)
            continue

        method_key = f"{method.replace('-', '_').lower()}_{dataset}"
        baseline_results[method_key] = {}
        n_biclass_total = 0
        for tc in (1, 2, 3):
            mask     = (test_classes_all == tc) & biclass_mask_all
            preds_c  = preds_all[mask]
            labels_c = labels_all[mask]
            valid    = ~np.isnan(preds_c)
            preds_c  = preds_c[valid]
            labels_c = labels_c[valid]
            n_biclass_total += len(preds_c)

            if len(preds_c) == 0 or len(np.unique(labels_c)) < 2:
                baseline_results[method_key][f"class_{tc}"] = {
                    "fprs": [], "tprs": [], "aucs": [], "ns": [0]
                }
                continue

            fpr, tpr, _ = roc_curve(labels_c, preds_c)
            score = auc(fpr, tpr)
            baseline_results[method_key][f"class_{tc}"] = {
                "fprs": [fpr], "tprs": [tpr], "aucs": [score], "ns": [len(preds_c)]
            }
        print(f"  {method}: biclass n={n_biclass_total:,} "
              f"(C3 AUC={baseline_results[method_key]['class_3']['aucs'] or 'n/a'})",
              flush=True)

    preds_file = os.path.join(WORKING_DIR, f"{dataset}_mutpred2_standalone_preds.npy")
    preds_all = load_positional_cache(preds_file, n_rows)
    if preds_all is not None:
        mask     = biclass_mask_all & ~np.isnan(preds_all) & (labels_all >= 0)
        preds_c  = preds_all[mask]
        labels_c = labels_all[mask]
        method_key = f"mutpred2_standalone_{dataset}"
        if len(preds_c) > 0 and len(np.unique(labels_c)) >= 2:
            fpr, tpr, _ = roc_curve(labels_c, preds_c)
            score = auc(fpr, tpr)
            baseline_results[method_key] = {
                f"class_{c}": {"fprs": [fpr], "tprs": [tpr], "aucs": [score],
                               "ns": [len(preds_c)]}
                for c in [1, 2, 3]
            }
            print(f"  mutpred2_standalone: biclass n={len(preds_c):,}  AUC={score:.3f}",
                  flush=True)

    return baseline_results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading biclass pairs from the canonical row table...", flush=True)
    biclass_pairs = load_biclass_pairs()
    canonical_rows = load_data(DATASET_CONFIGS[CANONICAL_DATASET])

    print("Building per-fold row identity from canonical rows + fold splits...", flush=True)
    row_ids_by_seed = build_row_ids_by_seed(canonical_rows, N_SEEDS)

    results_dict: dict = {}

    for method_key, pkl_name in METHODS:
        pkl_path = PKL_DIR / pkl_name
        if not pkl_path.exists():
            print(f"  [SKIP] {pkl_name} not found", flush=True)
            continue
        print(f"\nProcessing {method_key}...", flush=True)
        detailed_results = load_gcv_detailed_results(pkl_path, CANONICAL_DATASET)

        filtered = filter_detailed_results(detailed_results, row_ids_by_seed, biclass_pairs)
        roc      = compute_roc_with_variance(filtered)

        n_curves_c3 = len(roc["class_3"]["aucs"])
        if n_curves_c3 == 0:
            print(f"  [SKIP] No valid C3 curves after biclass filtering", flush=True)
            continue

        mean_c3 = float(np.mean(roc["class_3"]["aucs"]))
        std_c3  = float(np.std(roc["class_3"]["aucs"], ddof=1))
        print(f"  C3 AUROC: {mean_c3:.3f} ± {std_c3:.3f}  (n_curves={n_curves_c3})",
              flush=True)
        results_dict[method_key] = roc

    print(f"\nProcessing fixed-prediction baselines (SAAMBE-3D, MutPPI, MutPPI+, MutPred2)...",
          flush=True)
    complex_ids = load_pairs_in_row_order()
    baseline_results = filter_baseline_predictions(DATASET, biclass_pairs, complex_ids)
    results_dict.update(baseline_results)

    if not results_dict:
        print("ERROR: No methods produced valid results.", flush=True)
        return

    out_png = OUT_DIR / "roc_sahni_fragoza_biclass_with_variance.png"
    print(f"\nPlotting to {out_png}...", flush=True)
    plot_roc_with_confidence(
        results_dict,
        dataset_name="sahni_fragoza (biclass pairs)",
        save_path=str(out_png),
    )
    print(f"Saved → {out_png}", flush=True)

    # TSV summary of C3 AUROCs
    tsv_path = OUT_DIR / "biclass_c3_aurocs.tsv"
    with open(tsv_path, "w") as f:
        f.write("method\tdisplay_name\tmean_auc_c3\tstd_auc_c3\tn_curves\n")
        for mkey, roc in results_dict.items():
            aucs    = np.array(roc["class_3"]["aucs"])
            dname   = METHOD_DISPLAY_NAMES.get(mkey, mkey)
            f.write(f"{mkey}\t{dname}\t{np.mean(aucs):.4f}\t{np.std(aucs, ddof=1):.4f}"
                    f"\t{len(aucs)}\n")
    print(f"AUCs saved → {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
