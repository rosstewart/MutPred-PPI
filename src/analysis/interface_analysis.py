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
  results_revisions/robustness_analyses/interface_auroc_by_class.png
  results_revisions/robustness_analyses/interface_auroc_summary.tsv

Usage:
  conda run -n ppi python src/analysis/interface_analysis.py
"""

import os
import re
import pickle
import numpy as np
import scipy.io as sio
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
GRAPH_DIR = f"{_BASE}/2026/graphs"
GCV_RESULTS = f"{_PUB}/results_revisions/macro_aucs/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"
VT_IDS_FILE = f"{CV_DIR}/sahni_fragoza_train_all_vt_ids.pkl"
OUT_DIR = f"{_PUB}/results_revisions/robustness_analyses"
N_SEEDS = 30
MIN_N = 5  # matches roc_plots.py spirit: just require both label classes per fold
N_SEM_DIVISOR = 10  # matches hardcoded value in roc_plots.py compute_roc_with_variance
FPR_GRID = np.linspace(0, 1, 100)  # module-level: shared by compute_curves() and plot_on_axes()


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_variant_pos(variant: str) -> int:
    """Return 0-indexed residue position from e.g. 'M603V'."""
    m = re.search(r"(\d+)", variant)
    if not m:
        raise ValueError(f"Cannot parse position from variant: {variant}")
    return int(m.group(1)) - 1


def is_interface(graph_path: str, pos: int):
    """Return True if residue pos (0-indexed) makes cross-chain contacts.

    Returns None if the graph file is missing or the position is out of range.
    """
    if not os.path.exists(graph_path):
        return None
    try:
        mat = sio.loadmat(graph_path)
        G = mat["G"].toarray()
        NRR = int(mat["NRR"][0, 0])
        if pos >= NRR or pos < 0:
            return None
        inter_row = G[pos, NRR:]
        return bool(np.any(inter_row > 0))
    except Exception:
        return None


def complex_id_from_vt(vt_id: str) -> str:
    return vt_id.split(" ")[0]


def variant_from_vt(vt_id: str) -> str:
    return vt_id.split(" ")[1]


def build_interface_cache(vt_ids: list) -> dict:
    """Precompute interface status for every unique (complex_id, variant) pair."""
    cache = {}
    for vt_id in vt_ids:
        if vt_id in cache:
            continue
        cid = complex_id_from_vt(vt_id)
        variant = variant_from_vt(vt_id)
        graph_path = os.path.join(GRAPH_DIR, f"{cid}.mat")
        try:
            pos = parse_variant_pos(variant)
        except ValueError:
            cache[vt_id] = None
            continue
        cache[vt_id] = is_interface(graph_path, pos)
    return cache


# ── Main analysis ──────────────────────────────────────────────────────────────

def compute_curves():
    """Load data and compute per-(class, group) ROC fold curves.

    Returns (fold_curves, all_vt_ids_group) — same structure previously built
    inline in main(), now reusable by combined_robustness_figure.py.
    """
    with open(VT_IDS_FILE, "rb") as f:
        all_vt_ids = pickle.load(f)
    print(f"Loaded {len(all_vt_ids)} vt_ids")

    print("Building interface label cache...")
    iface_cache = build_interface_cache(all_vt_ids)
    n_iface = sum(1 for v in iface_cache.values() if v is True)
    n_no_graph = sum(1 for v in iface_cache.values() if v is None)
    print(f"  Interface: {n_iface}, Non-interface: {len(iface_cache)-n_iface-n_no_graph}, Missing graphs: {n_no_graph}")

    with open(GCV_RESULTS, "rb") as f:
        gcv_results = pickle.load(f)


    # fold_curves[class][group] = list of per-fold interpolated TPR arrays
    fold_curves = {c: {"interface": [], "non_interface": []} for c in (1, 2, 3)}
    all_vt_ids_group = {c: {"interface": set(), "non_interface": set()} for c in (1, 2, 3)}

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

            # vt_ids for this fold's test samples
            fold_vt_ids = [vt_ids_seed[idx] for idx in test_idx]

            # Reconstruct preds/labels in test-sample order
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
            iface_flags = np.array([iface_cache.get(vt) for vt in fold_vt_ids], dtype=object)

            for cls in (1, 2, 3):
                mask_cls = classes_ordered == cls
                for is_iface, key in [(True, "interface"), (False, "non_interface")]:
                    mask = mask_cls & (iface_flags == is_iface)
                    p = preds_ordered[mask]
                    l = labels_ordered[mask]
                    vts = np.array(fold_vt_ids)[mask]
                    all_vt_ids_group[cls][key].update(vts)
                    if len(p) >= MIN_N and len(np.unique(l)) == 2:
                        fpr, tpr, _ = roc_curve(l, p)
                        fold_curves[cls][key].append(np.interp(FPR_GRID, fpr, tpr))

            flat_cursor += n_test

        print(f"  Seed {seed}: done", flush=True)

    return fold_curves, all_vt_ids_group


CLASS_LABELS = {1: "Class 1 (both seen)", 2: "Class 2 (one seen)", 3: "Class 3 (neither seen)"}
CURVE_COLORS = {"interface": "#1f77b4", "non_interface": "#aec7e8"}
DISPLAY_NAMES = {"interface": "Interface", "non_interface": "Non-interface"}


def plot_on_axes(axes, fold_curves, all_vt_ids_group):
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
            n_variants = len(all_vt_ids_group[cls][key])
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
    fold_curves, all_vt_ids_group = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_vt_ids_group)

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
