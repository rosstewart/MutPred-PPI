#!/usr/bin/env python
"""Single-domain vs multi-domain stratification for Reviewer 1.

Classifies each test interactor (the protein carrying the mutation) as
single-domain (1 unique InterPro domain family) or multi-domain (2+), using
the pre-built pfam_domains_cache.pkl. Computes AUROC separately per group
across 30 seeds × 10 folds of existing GCV test predictions — no new GCV run.

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10)

Output:
  results_revisions/robustness_analyses/protein_class_auroc_by_class.png
  results_revisions/robustness_analyses/protein_class_auroc_summary.tsv
  figures/protein_class_auroc_by_class.png  (symlink)

Usage:
  conda run -n ppi python src/analysis/protein_class_stratification.py
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
from paths import ANNOTATIONS_DIR, DATA_ROOT, REPO_ROOT, cv_reference_dir


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
PFAM_CACHE = str(ANNOTATIONS_DIR / "pfam_domains_cache.pkl")
GCV_RESULTS = f"{_PUB}/results_revisions/macro_aucs/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"
VT_IDS_FILE = f"{CV_DIR}/sahni_fragoza_train_all_vt_ids.pkl"
OUT_DIR   = f"{_PUB}/results_revisions/robustness_analyses"

N_SEEDS       = 30
MIN_N         = 5
from gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)

GROUPS  = ["single", "multi"]
COLORS  = {"single": "#1a9641", "multi": "#a6d96a"}
LABELS  = {"single": "Single-domain", "multi": "Multi-domain"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_domain_lookup(pfam_hits: dict) -> dict:
    """UniProt ID → 'single' | 'multi' | None."""
    out = {}
    for uid, ipr_list in pfam_hits.items():
        n = len(set(ipr_list))
        out[uid] = "single" if n == 1 else "multi"
    return out


def build_vt_group_cache(vt_ids: list, domain_lookup: dict) -> dict:
    """Map vt_id → group (or None if interactor not in cache)."""
    seen = {}
    result = {}
    for vt_id in vt_ids:
        complex_id = vt_id.split(" ")[0]
        if complex_id not in seen:
            interactor = complex_id.split("-")[0]
            seen[complex_id] = domain_lookup.get(interactor)
        result[vt_id] = seen[complex_id]
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def compute_curves():
    """Load data and compute per-(class, domain-group) ROC fold curves.

    Returns (fold_curves, all_vt_by_grp).
    """
    print("Loading pfam_domains_cache...", flush=True)
    with open(PFAM_CACHE, "rb") as f:
        pfam = pickle.load(f)
    domain_lookup = build_domain_lookup(pfam["hits"])
    print(f"  {len(domain_lookup)} proteins with IPR hits", flush=True)

    with open(VT_IDS_FILE, "rb") as f:
        all_vt_ids = pickle.load(f)

    vt_groups = build_vt_group_cache(all_vt_ids, domain_lookup)
    for g in GROUPS:
        print(f"  {g}: {sum(1 for v in vt_groups.values() if v == g)}", flush=True)
    print(f"  unknown: {sum(1 for v in vt_groups.values() if v is None)}", flush=True)
    total = len(all_vt_ids)
    known = sum(1 for v in vt_groups.values() if v is not None)
    print(f"  Coverage: {known}/{total} = {known/total:.1%}", flush=True)

    with open(GCV_RESULTS, "rb") as f:
        gcv_results = pickle.load(f)

    fold_curves   = {c: {g: [] for g in GROUPS} for c in (1, 2, 3)}
    all_vt_by_grp = {c: {g: set() for g in GROUPS} for c in (1, 2, 3)}

    for seed in range(N_SEEDS):
        vt_ids_seed_path = f"{CV_DIR}/sahni_fragoza_train_all_vt_ids_{seed}.pkl"
        fold_splits_path = f"{CV_DIR}/sahni_fragoza_train_fold_splits_{seed}.pkl"
        ptc_path         = f"{CV_DIR}/swing_train_pair_test_classes_{seed}.npy"

        if not all(os.path.exists(p) for p in [vt_ids_seed_path, fold_splits_path, ptc_path]):
            print(f"  Seed {seed}: missing files, skipping", flush=True)
            continue

        with open(vt_ids_seed_path, "rb") as f:
            vt_ids_seed = pickle.load(f)
        with open(fold_splits_path, "rb") as f:
            fold_splits = pickle.load(f)
        pair_test_classes = np.load(ptc_path)

        iteration   = gcv_results["iterations"][seed]
        flat_cursor = 0

        for fold_tuple in sorted(fold_splits, key=lambda t: t[0]):
            fold, train_idx, test_idx = fold_tuple
            fold_data = iteration["folds"][fold]
            n_test    = len(test_idx)
            ptc_fold  = pair_test_classes[flat_cursor:flat_cursor + n_test]
            fold_vt_ids = [vt_ids_seed[idx] for idx in test_idx]

            preds_fold  = {cls: list(fold_data[f"class_{cls}"]["preds"])  for cls in (1, 2, 3)}
            labels_fold = {cls: list(fold_data[f"class_{cls}"]["labels"]) for cls in (1, 2, 3)}
            cls_cursor  = {1: 0, 2: 0, 3: 0}
            preds_ordered, labels_ordered = [], []
            for cls in ptc_fold:
                preds_ordered.append(preds_fold[cls][cls_cursor[cls]])
                labels_ordered.append(labels_fold[cls][cls_cursor[cls]])
                cls_cursor[cls] += 1

            preds_ordered  = np.array(preds_ordered)
            labels_ordered = np.array(labels_ordered)
            classes_ordered = np.array(ptc_fold)
            groups_ordered  = np.array([vt_groups.get(vt) for vt in fold_vt_ids], dtype=object)

            for cls in (1, 2, 3):
                mask_cls = classes_ordered == cls
                for grp in GROUPS:
                    mask = mask_cls & (groups_ordered == grp)
                    p = preds_ordered[mask]
                    l = labels_ordered[mask]
                    vts = np.array(fold_vt_ids)[mask]
                    all_vt_by_grp[cls][grp].update(vts)
                    if len(p) >= MIN_N and len(np.unique(l)) == 2:
                        fpr, tpr, _ = roc_curve(l, p)
                        fold_curves[cls][grp].append(np.interp(FPR_GRID, fpr, tpr))

            flat_cursor += n_test

        print(f"  Seed {seed}: done", flush=True)

    return fold_curves, all_vt_by_grp


CLASS_LABELS = {1: "Class 1 (both seen)", 2: "Class 2 (one seen)", 3: "Class 3 (neither seen)"}


def plot_on_axes(axes, fold_curves, all_vt_by_grp):
    """Draw the 3-panel (C1/C2/C3) single-vs-multi-domain ROC comparison onto
    pre-supplied axes. Returns summary_rows (list[str]) for the TSV output.
    """
    class_labels = CLASS_LABELS

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        plot_entries = []

        for grp in GROUPS:
            curves      = fold_curves[cls][grp]
            n_fold      = len(curves)
            n_variants  = len(all_vt_by_grp[cls][grp])
            color       = COLORS[grp]

            if curves:
                arr      = np.array(curves)
                mean_tpr = np.mean(arr, axis=0)
                std_tpr  = np.std(arr, axis=0, ddof=1)
                sem_tpr  = std_tpr / np.sqrt(N_SEM_DIVISOR)
                per_fold_aucs = [np.trapz(c, FPR_GRID) for c in curves]
                mean_auc = float(np.mean(per_fold_aucs))
                sem_auc  = float(np.std(per_fold_aucs, ddof=1) / np.sqrt(N_SEM_DIVISOR))
                plot_entries.append((mean_auc, grp, color, mean_tpr,
                                     mean_tpr - sem_tpr, mean_tpr + sem_tpr,
                                     sem_auc, n_variants, n_fold))
                summary_rows.append(
                    f"C{cls}\t{grp}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}\t{n_fold}"
                )
            else:
                summary_rows.append(f"C{cls}\t{grp}\tnan\tnan\t{n_variants}\t0")

        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, grp, color, mean_tpr, lo_tpr, hi_tpr, sem_auc, n_variants, _ in plot_entries:
            label = f"{LABELS[grp]} (n={n_variants:,}, AUC={mean_auc:.3f}±{sem_auc:.3f})"
            ax.plot(FPR_GRID, mean_tpr, color=color, lw=2, label=label)
            ax.fill_between(FPR_GRID, lo_tpr, hi_tpr, color=color, alpha=0.15)

        ax.set_title(class_labels[cls], fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        if cls == 1:
            ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)

    return summary_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fold_curves, all_vt_by_grp = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_vt_by_grp)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "protein_class_auroc_by_class.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}", flush=True)

    out_tsv = os.path.join(OUT_DIR, "protein_class_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("class\tgroup\tmean_auroc\tsem\tn_variants\tn_fold_curves\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}", flush=True)

    # Symlink into figures/
    fig_link = f"{_PUB}/figures/protein_class_auroc_by_class.png"
    if os.path.islink(fig_link):
        os.remove(fig_link)
    os.symlink(out_png, fig_link)
    print(f"Symlink: {fig_link}", flush=True)


if __name__ == "__main__":
    main()
