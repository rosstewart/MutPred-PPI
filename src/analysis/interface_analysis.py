#!/usr/bin/env python
"""Interface vs. non-interface variant analysis for Reviewer 1.

For each of 30 GCV seeds × 10 folds = up to 300 per-fold ROC curves,
reconstructs which test samples are at the PPI interface (the mutated residue
makes at least one cross-chain contact in the AF3 contact graph), then computes
AUROC separately for interface and non-interface variants across the three test
classes (C1/C2/C3).

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10) (hardcoded denominator)

Output:
  results/robustness/interface_auroc_by_class.png
  results/robustness/interface_auroc_summary.tsv

Usage:
  conda run -n ppi python src/analysis/interface_analysis.py
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import DATA_ROOT, REPO_ROOT, cv_reference_dir
from variant_db_inference import variant_rows as vr
from utils.gcv_common import StaleCacheError, load_gcv_detailed_results  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
TRAIN_EVAL_STORE = f"{_PUB}/datasets/training_eval/contact_graphs.h5"
GCV_RESULTS = f"{_PUB}/results/gcv/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"
CANONICAL_DATASET = "sahni_fragoza_mapped090826"
CANONICAL_ROWS_PATH = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR = f"{_PUB}/results/robustness"
N_SEEDS = 30
MIN_N = 5  # matches roc_plots.py spirit: just require both label classes per fold
from gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mutation_pos1(mutation_1b: str) -> int:
    """Return the 1-based residue position from a canonical mutation string like 'M603V'.

    Canonical mutations in the row tables are 1-based. This is the one place the
    position is extracted, so the conversion is explicit rather than distributed.
    """
    if not re.search(r"\d", mutation_1b):
        raise ValueError(f"Cannot parse position from mutation: {mutation_1b!r}")
    return int(re.search(r"(\d+)", mutation_1b).group(1))


def is_interface(store, interactor_seq: str, partner_seq: str, pos1: int):
    """True if residue `pos1` (1-BASED, in the INTERACTOR) contacts the partner.

    Resolved by sequence content, not by filename. The store returns the graph
    already oriented to the requested interactor, so the interactor always
    occupies rows `[0, len(interactor_seq))` and the partner the rest — whereas
    the previous `.mat` path read `NRR` and *assumed* chain A was the interactor,
    which is false for the 121 clinvar / 121 cosmic / 116 gnomad / 29 hgmd graphs
    whose stored chain order contradicts their own filename.

    Returns None when the pair has no graph or the position is out of range.
    """
    if pos1 < 1 or pos1 > len(interactor_seq):
        return None
    G = store.load_dense(interactor=interactor_seq, partner=partner_seq)
    if G is None:                       # pair absent from the store
        return None
    n_inter = len(interactor_seq)
    row = pos1 - 1          # the one 1-based -> array-index conversion
    return bool(np.any(G[row, n_inter:] > 0))


def _complex_id_sequences(dataset: str = "sahni_fragoza_mapped090826") -> dict:
    """`'{interactor}-{partner}'` -> (interactor_sequence, partner_sequence).

    The key is CONSTRUCTED from the canonical table's own columns and matched
    whole. It is never produced by splitting a `complex_id` on `-`: 261 of the
    2,785 complex_ids in the CV reference contain more than one `-` (e.g.
    `O43889-2-J3QKU0`, an isoform accession), so a split silently mis-assigns
    both accessions for ~9% of pairs.
    """
    from utils.gcv_common import DATASET_CONFIGS, load_data
    df = load_data(DATASET_CONFIGS[dataset])
    return {f"{i}-{p}": (a, b)
            for i, p, a, b in zip(df["interactor"], df["partner"],
                                  df["interactor_sequence"], df["partner_sequence"])}


def build_row_iface_flags(canonical_rows: pd.DataFrame, store=None) -> list:
    """Per-canonical-row interface flag (True/False/None), in `row_index` order.

    Returns a list of length `len(canonical_rows)` where `result[i]` is the
    interface status of row `i`.  The pair sequences come from the canonical
    table's own columns (no dict lookup / no split), so there is no split-on-`-`
    and no coverage gap from ambiguous complex_ids.

    Replacing the old `vt_ids`-keyed cache removes the dependency on the
    pre-090826 5,894-row vt_ids pickles.
    """
    from contact_graphs import ContactGraphStore

    seq_map = _complex_id_sequences()
    own_store = store is None
    if own_store:
        store = ContactGraphStore(TRAIN_EVAL_STORE)

    result: list[bool | None] = []
    no_seq = no_graph = 0
    try:
        for _, row in canonical_rows.iterrows():
            interactor, partner = str(row["interactor"]), str(row["partner"])
            mutation = str(row["mutation"])
            seqs = seq_map.get(f"{interactor}-{partner}")
            if seqs is None:
                no_seq += 1
                result.append(None)
                continue
            try:
                pos1 = _mutation_pos1(mutation)
            except ValueError:
                result.append(None)
                continue
            flag = is_interface(store, seqs[0], seqs[1], pos1)
            if flag is None:
                no_graph += 1
            result.append(flag)
    finally:
        if own_store:
            store.close()

    if no_seq:
        print(f"  {no_seq} rows: pair absent from canonical table (sequences missing)")
    if no_graph:
        print(f"  {no_graph} rows: pair resolved but absent from the contact-graph store")
    return result


# ── Main analysis ──────────────────────────────────────────────────────────────

def compute_curves():
    """Load data and compute per-(class, group) ROC fold curves.

    Returns (fold_curves, all_rows_group) — same structure as before but keyed by
    row_index rather than vt_id strings.  Row identity is the positional integer
    into the 6,219-row canonical table (`sahni_fragoza_train_rows.csv.gz`).

    This removes the dependency on the pre-090826 5,894-row vt_ids pickles.
    """
    # ── Canonical rows ────────────────────────────────────────────────────────
    canonical_rows = pd.read_csv(CANONICAL_ROWS_PATH)
    n_rows = len(canonical_rows)
    print(f"Canonical rows: {n_rows}")

    # ── Per-row interface flags ────────────────────────────────────────────────
    print("Building interface label cache from canonical rows...")
    row_iface = build_row_iface_flags(canonical_rows)  # list[bool|None], len=n_rows
    n_iface     = sum(1 for v in row_iface if v is True)
    n_non_iface = sum(1 for v in row_iface if v is False)
    n_no_graph  = sum(1 for v in row_iface if v is None)
    print(f"  Interface: {n_iface}, Non-interface: {n_non_iface}, Missing: {n_no_graph}")

    # ── GCV results ───────────────────────────────────────────────────────────
    gcv_results = load_gcv_detailed_results(GCV_RESULTS, CANONICAL_DATASET)

    # fold_curves[class][group] = list of per-fold interpolated TPR arrays
    fold_curves = {c: {"interface": [], "non_interface": []} for c in (1, 2, 3)}
    # all_rows_group[class][group] = set of row_index values (for n_variants counting)
    all_rows_group = {c: {"interface": set(), "non_interface": set()} for c in (1, 2, 3)}

    for seed in range(N_SEEDS):
        fold_splits_path = f"{CV_DIR}/sahni_fragoza_train_fold_splits_{seed}.pkl"
        ptc_path = f"{CV_DIR}/swing_train_pair_test_classes_{seed}.npy"

        if not all(os.path.exists(p) for p in [fold_splits_path, ptc_path]):
            print(f"  Seed {seed}: missing fold_splits or ptc, skipping")
            continue

        with open(fold_splits_path, "rb") as f:
            fold_splits = pickle.load(f)
        pair_test_classes = np.load(ptc_path)

        iteration = gcv_results["iterations"].get(seed)
        if iteration is None:
            print(f"  Seed {seed}: not in GCV results, skipping")
            continue

        # Freshness of GCV_RESULTS as a whole is already asserted once, by
        # `load_gcv_detailed_results` above (via the shared
        # `utils.gcv_common.assert_gcv_pkl_fresh`) -- every seed in the same
        # pkl was written by the same run, so a per-seed re-check here would
        # only repeat that one assertion, not add coverage.

        flat_cursor = 0
        for fold_tuple in sorted(fold_splits, key=lambda t: t[0]):
            fold, train_idx, test_idx = fold_tuple
            fold_data = iteration["folds"][fold]
            n_test = len(test_idx)
            ptc_fold = pair_test_classes[flat_cursor:flat_cursor + n_test]

            # Per-row interface flags for this fold's test rows.
            fold_iface = [row_iface[ridx] for ridx in test_idx]

            # Reconstruct preds/labels in test-sample order.
            preds_fold = {cls: list(fold_data[f"class_{cls}"]["preds"]) for cls in (1, 2, 3)}
            labels_fold = {cls: list(fold_data[f"class_{cls}"]["labels"]) for cls in (1, 2, 3)}

            # The overall pkl's row count matched the canonical table (checked
            # once, above), but that does not guarantee THIS fold's per-class
            # buckets line up with THIS fold split -- the interleave below
            # indexes `preds_fold[cls][cls_cursor[cls]]` unconditionally, so a
            # per-fold mismatch must be caught here or it surfaces as an opaque
            # `IndexError` instead of a named cause.
            n_cached = sum(len(v) for v in preds_fold.values())
            if n_cached != n_test:
                raise StaleCacheError(
                    f"{GCV_RESULTS}: seed {seed} fold {fold} holds {n_cached} "
                    f"cached predictions but the canonical fold has {n_test} "
                    f"test rows, despite the pkl's overall row count matching "
                    f"the canonical table. This fold is internally inconsistent "
                    f"and must be recomputed.")

            preds_ordered, labels_ordered = [], []
            cls_cursor = {1: 0, 2: 0, 3: 0}
            for cls in ptc_fold:
                preds_ordered.append(preds_fold[cls][cls_cursor[cls]])
                labels_ordered.append(labels_fold[cls][cls_cursor[cls]])
                cls_cursor[cls] += 1

            preds_ordered = np.array(preds_ordered)
            labels_ordered = np.array(labels_ordered)
            classes_ordered = np.array(ptc_fold)
            iface_flags = np.array(fold_iface, dtype=object)

            for cls in (1, 2, 3):
                mask_cls = classes_ordered == cls
                for is_iface, key in [(True, "interface"), (False, "non_interface")]:
                    mask = mask_cls & (iface_flags == is_iface)
                    p = preds_ordered[mask]
                    l = labels_ordered[mask]
                    ridxs = np.array(test_idx)[mask]
                    all_rows_group[cls][key].update(ridxs.tolist())
                    if len(p) >= MIN_N and len(np.unique(l)) == 2:
                        fpr, tpr, _ = roc_curve(l, p)
                        fold_curves[cls][key].append(np.interp(FPR_GRID, fpr, tpr))

            flat_cursor += n_test

        print(f"  Seed {seed}: done", flush=True)

    return fold_curves, all_rows_group


CLASS_LABELS = {1: "Class 1 (both seen)", 2: "Class 2 (one seen)", 3: "Class 3 (neither seen)"}
CURVE_COLORS = {"interface": "#1f77b4", "non_interface": "#aec7e8"}
DISPLAY_NAMES = {"interface": "Interface", "non_interface": "Non-interface"}


def plot_on_axes(axes, fold_curves, all_rows_group):
    """Draw the 3-panel (C1/C2/C3) ROC comparison onto pre-supplied axes.

    Returns summary_rows (list[str]) for the TSV output.
    """
    class_labels = CLASS_LABELS
    curve_colors = CURVE_COLORS
    display_names = DISPLAY_NAMES

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        # Compute all curves first, then sort legend by AUC descending
        plot_entries = []
        for key in ["interface", "non_interface"]:
            curves = fold_curves[cls][key]
            n_fold_curves = len(curves)
            n_variants = len(all_rows_group[cls][key])
            color = curve_colors[key]

            if curves:
                arr = np.array(curves)
                mean_tpr = np.mean(arr, axis=0)
                std_tpr = np.std(arr, axis=0, ddof=1)
                sem_tpr = std_tpr / np.sqrt(N_SEM_DIVISOR)
                lo_tpr = mean_tpr - sem_tpr
                hi_tpr = mean_tpr + sem_tpr
                per_fold_aucs = [np.trapz(c, FPR_GRID) for c in curves]
                mean_auc = float(np.mean(per_fold_aucs))
                sem_auc = float(np.std(per_fold_aucs, ddof=1) / np.sqrt(N_SEM_DIVISOR))
                plot_entries.append((mean_auc, key, color, mean_tpr, lo_tpr, hi_tpr,
                                     sem_auc, n_variants, n_fold_curves))
                summary_rows.append(
                    f"C{cls}\t{key}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}\t{n_fold_curves}"
                )
            else:
                summary_rows.append(f"C{cls}\t{key}\tnan\tnan\t{n_variants}\t0")

        # Sort descending by AUC before plotting (so legend is AUC-sorted)
        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, key, color, mean_tpr, lo_tpr, hi_tpr, sem_auc, n_variants, _ in plot_entries:
            label = f"{display_names[key]} (n={n_variants:,}, AUC={mean_auc:.3f}±{sem_auc:.3f})"
            ax.plot(FPR_GRID, mean_tpr, color=color, lw=2, label=label)
            ax.fill_between(FPR_GRID, lo_tpr, hi_tpr, color=color, alpha=0.15)

        ax.set_title(class_labels[cls], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        if cls == 1:
            ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

    return summary_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fold_curves, all_rows_group = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_rows_group)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "interface_auroc_by_class.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "interface_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("class\tgroup\tmean_auroc\tsem\tn_variants\tn_fold_curves\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
