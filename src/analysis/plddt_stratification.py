#!/usr/bin/env python
"""pLDDT quality stratification for Reviewer 1.

Bins test complexes by mean AF3 pLDDT confidence (complex-level average CA
B-factor across both chains), then computes AUROC separately for each bin
across 30 seeds × 10 folds = up to 300 per-fold ROC curves.

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10) (hardcoded denominator)

Confidence bins (per complex):
  Low:    mean pLDDT < 70
  Medium: 70 ≤ mean pLDDT < 85
  High:   mean pLDDT ≥ 85

Output:
  results_revisions/robustness_analyses/plddt_auroc_by_class.png
  results_revisions/robustness_analyses/plddt_auroc_summary.tsv

Usage:
  conda run -n ppi python src/analysis/plddt_stratification.py
"""

import os
import pickle
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CV_DIR as _P_CV_DIR, DATA_ROOT, REPO_ROOT, cv_reference_dir  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
PLDDT_CACHE = f"{_BASE}/2026/plddt_cache.pkl"
GCV_RESULTS = f"{_PUB}/results_revisions/macro_aucs/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"
VT_IDS_FILE = f"{CV_DIR}/sahni_fragoza_train_all_vt_ids.pkl"
OUT_DIR = f"{_PUB}/results_revisions/robustness_analyses"
N_SEEDS = 30
MIN_N = 5  # matches roc_plots.py spirit: just require both label classes per fold
N_SEM_DIVISOR = 10  # matches hardcoded value in roc_plots.py compute_roc_with_variance
FPR_GRID = np.linspace(0, 1, 100)  # module-level: shared by compute_curves() and plot_on_axes()

PLDDT_BINS = [
    ("low",    0,    70),
    ("medium", 70,   85),
    ("high",   85,  100),
]
BIN_COLORS = {"low": "#d6604d", "medium": "#f4a582", "high": "#4393c3"}
BIN_LABELS = {"low": "Low (<70)", "medium": "Medium (70–85)", "high": "High (≥85)"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def complex_mean_plddt(complex_id: str, plddt_cache: dict):
    """Return mean pLDDT for both chains in the complex, or None if missing."""
    parts = complex_id.split("-")
    if len(parts) < 2:
        return None
    prot1, prot2 = parts[0], "-".join(parts[1:])
    entry1 = plddt_cache.get(prot1)
    entry2 = plddt_cache.get(prot2)
    if entry1 is None or entry2 is None:
        return None
    arr1 = entry1["plddt"] if isinstance(entry1, dict) else entry1
    arr2 = entry2["plddt"] if isinstance(entry2, dict) else entry2
    return float(np.mean(np.concatenate([arr1, arr2])))


def bin_plddt(val: float) -> str:
    for name, lo, hi in PLDDT_BINS:
        if lo <= val < hi:
            return name
    return "high"


def build_plddt_bin_cache(vt_ids: list, plddt_cache: dict) -> dict:
    """Map each vt_id to its pLDDT bin (or None)."""
    seen = {}
    result = {}
    for vt_id in vt_ids:
        cid = vt_id.split(" ")[0]
        if cid not in seen:
            mean_val = complex_mean_plddt(cid, plddt_cache)
            seen[cid] = bin_plddt(mean_val) if mean_val is not None else None
        result[vt_id] = seen[cid]
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def compute_curves():
    """Load data and compute per-(class, pLDDT bin) ROC fold curves.

    Returns (fold_curves, all_vt_ids_bin).
    """
    print("Loading pLDDT cache...")
    with open(PLDDT_CACHE, "rb") as f:
        plddt_cache = pickle.load(f)
    print(f"  {len(plddt_cache)} proteins in cache")

    with open(VT_IDS_FILE, "rb") as f:
        all_vt_ids = pickle.load(f)

    plddt_bins = build_plddt_bin_cache(all_vt_ids, plddt_cache)
    bin_counts = {b: sum(1 for v in plddt_bins.values() if v == b) for b, *_ in PLDDT_BINS}
    none_count = sum(1 for v in plddt_bins.values() if v is None)
    print(f"  pLDDT bins: {bin_counts} | missing: {none_count}")

    with open(GCV_RESULTS, "rb") as f:
        gcv_results = pickle.load(f)


    # fold_curves[class][bin] = list of per-fold interpolated TPR arrays
    fold_curves = {c: {b: [] for b, *_ in PLDDT_BINS} for c in (1, 2, 3)}
    all_vt_ids_bin = {c: {b: set() for b, *_ in PLDDT_BINS} for c in (1, 2, 3)}

    for seed in range(N_SEEDS):
        vt_ids_seed_path = f"{CV_DIR}/sahni_fragoza_train_all_vt_ids_{seed}.pkl"
        fold_splits_path = f"{CV_DIR}/sahni_fragoza_train_fold_splits_{seed}.pkl"
        ptc_path = f"{CV_DIR}/swing_train_pair_test_classes_{seed}.npy"

        if not all(os.path.exists(p) for p in [vt_ids_seed_path, fold_splits_path, ptc_path]):
            print(f"  Seed {seed}: missing files, skipping")
            continue

        with open(vt_ids_seed_path, "rb") as f:
            vt_ids_seed = pickle.load(f)
        with open(fold_splits_path, "rb") as f:
            fold_splits = pickle.load(f)
        pair_test_classes = np.load(ptc_path)

        iteration = gcv_results["iterations"][seed]

        flat_cursor = 0
        for fold_tuple in sorted(fold_splits, key=lambda t: t[0]):
            fold, train_idx, test_idx = fold_tuple
            fold_data = iteration["folds"][fold]
            n_test = len(test_idx)
            ptc_fold = pair_test_classes[flat_cursor:flat_cursor + n_test]

            fold_vt_ids = [vt_ids_seed[idx] for idx in test_idx]

            preds_fold = {cls: list(fold_data[f"class_{cls}"]["preds"]) for cls in (1, 2, 3)}
            labels_fold = {cls: list(fold_data[f"class_{cls}"]["labels"]) for cls in (1, 2, 3)}
            preds_ordered, labels_ordered = [], []
            cls_cursor = {1: 0, 2: 0, 3: 0}
            for cls in ptc_fold:
                preds_ordered.append(preds_fold[cls][cls_cursor[cls]])
                labels_ordered.append(labels_fold[cls][cls_cursor[cls]])
                cls_cursor[cls] += 1

            preds_ordered = np.array(preds_ordered)
            labels_ordered = np.array(labels_ordered)
            classes_ordered = np.array(ptc_fold)
            bins_by_order = np.array([plddt_bins.get(vt) for vt in fold_vt_ids], dtype=object)

            for cls in (1, 2, 3):
                mask_cls = classes_ordered == cls
                for bin_name, lo, hi in PLDDT_BINS:
                    mask = mask_cls & (bins_by_order == bin_name)
                    p = preds_ordered[mask]
                    l = labels_ordered[mask]
                    vts = np.array(fold_vt_ids)[mask]
                    all_vt_ids_bin[cls][bin_name].update(vts)
                    if len(p) >= MIN_N and len(np.unique(l)) == 2:
                        fpr, tpr, _ = roc_curve(l, p)
                        fold_curves[cls][bin_name].append(np.interp(FPR_GRID, fpr, tpr))

            flat_cursor += n_test

        print(f"  Seed {seed}: done", flush=True)

    return fold_curves, all_vt_ids_bin


CLASS_LABELS = {1: "Class 1 (both seen)", 2: "Class 2 (one seen)", 3: "Class 3 (neither seen)"}


def plot_on_axes(axes, fold_curves, all_vt_ids_bin):
    """Draw the 3-panel (C1/C2/C3) pLDDT-bin ROC comparison onto pre-supplied axes.

    Returns summary_rows (list[str]) for the TSV output.
    """
    bin_names = [b for b, *_ in PLDDT_BINS]
    class_labels = CLASS_LABELS

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        # Compute all curves first, then sort legend by AUC descending
        plot_entries = []
        for bin_name in bin_names:
            curves = fold_curves[cls][bin_name]
            n_fold_curves = len(curves)
            n_variants = len(all_vt_ids_bin[cls][bin_name])
            color = BIN_COLORS[bin_name]

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
                plot_entries.append((mean_auc, bin_name, color, mean_tpr, lo_tpr, hi_tpr,
                                     sem_auc, n_variants, n_fold_curves))
                summary_rows.append(
                    f"C{cls}\t{bin_name}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}\t{n_fold_curves}"
                )
            else:
                summary_rows.append(f"C{cls}\t{bin_name}\tnan\tnan\t{n_variants}\t0")

        # Sort descending by AUC before plotting (so legend is AUC-sorted)
        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, bin_name, color, mean_tpr, lo_tpr, hi_tpr, sem_auc, n_variants, _ in plot_entries:
            label = f"{BIN_LABELS[bin_name]} (n={n_variants:,}, AUC={mean_auc:.3f}±{sem_auc:.3f})"
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
    fold_curves, all_vt_ids_bin = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_vt_ids_bin)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "plddt_auroc_by_class.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "plddt_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("class\tplddt_bin\tmean_auroc\tsem\tn_variants\tn_fold_curves\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
