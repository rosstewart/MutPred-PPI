#!/usr/bin/env python
"""Single-domain vs multi-domain stratification for Reviewer 1.

Classifies each test interactor (the protein carrying the mutation) as
single-domain (1 unique InterPro domain family) or multi-domain (2+), using
the pre-built pfam_domains_cache.pkl. Computes AUROC separately per group
across 30 seeds × 10 folds of existing GCV test predictions — no new GCV run.

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10)

Rows come from the CANONICAL table `sahni_fragoza_train_rows.csv.gz`, whose
`row_index` is what `sahni_fragoza_train_fold_splits_{seed}.pkl` and
`swing_train_pair_test_classes_{seed}.npy` index. The previous version indexed a
6,219-entry split into a 5,894-entry `all_vt_ids_{seed}.pkl` and recovered the
mutated protein as `complex_id.split("-")[0]`, which is the wrong protein
whenever the interactor carries an isoform suffix. The table has `interactor` as
a column, so neither step exists any more.

Output:
  results/robustness/protein_class_auroc_by_class.png
  results/robustness/protein_class_auroc_summary.tsv
  figures/protein_class_auroc_by_class.png  (symlink)

Usage:
  conda run -n ppi python src/analysis/protein_class_stratification.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, DATA_ROOT, REPO_ROOT, cv_reference_dir
from analysis.stratification_common import (  # noqa: E402
    load_canonical_rows, stratified_fold_curves)
from utils.identifiers import bare_accession  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
PFAM_CACHE = str(ANNOTATIONS_DIR / "pfam_domains_cache.pkl")
GCV_RESULTS = f"{_PUB}/results/gcv/MutPredPPI_sahni_fragoza_megascale_all_detailed_results.pkl"
CANONICAL_DATASET = "sahni_fragoza_mapped090826"
# Canonical row ordering: row_index indexes the fold splits and test classes.
ROWS_FILE = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR   = f"{_PUB}/results/robustness"

# `StaleGcvCacheError` (a locally-defined duplicate of `StaleCacheError`)
# retired 2026-09-10: `StaleCacheError` from `utils.gcv_common` is the one
# exception every GCV-pkl consumer raises, so a script-specific name for the
# same failure no longer serves a purpose.

N_SEEDS       = 30
MIN_N         = 5
from analysis.gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)

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


def build_row_groups(rows: pd.DataFrame, domain_lookup: dict) -> np.ndarray:
    """Domain group per canonical row, positionally aligned to `row_index`.

    `bare_accession` is called explicitly because `pfam_domains_cache.pkl` is
    keyed by parent accessions only -- it has no isoform entries at all, so an
    isoform interactor would otherwise be scored as "unknown" rather than
    inheriting its parent's domain architecture. That is the one thing the
    suffix is dropped for; the row itself keeps the accession the table stores.
    """
    return np.array(
        [domain_lookup.get(bare_accession(a)) for a in rows["interactor"]],
        dtype=object,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def compute_curves():
    """Per-(test class, domain-group) ROC fold curves.

    The reconstruction itself lives in `analysis.stratification_common`, shared
    with the pLDDT and interface supplements; this only builds the grouping.
    """
    print("Loading pfam_domains_cache...", flush=True)
    with open(PFAM_CACHE, "rb") as f:
        pfam = pickle.load(f)
    domain_lookup = build_domain_lookup(pfam["hits"])
    print(f"  {len(domain_lookup)} proteins with IPR hits", flush=True)

    rows = load_canonical_rows(ROWS_FILE)
    print(f"  {len(rows)} canonical rows", flush=True)

    row_groups = build_row_groups(rows, domain_lookup)
    for g in GROUPS:
        print(f"  {g}: {int((row_groups == g).sum())}", flush=True)
    unknown = int(sum(1 for v in row_groups if v is None))
    print(f"  unknown: {unknown}", flush=True)
    known = len(row_groups) - unknown
    print(f"  Coverage: {known}/{len(rows)} = {known/len(rows):.1%}", flush=True)

    return stratified_fold_curves(row_groups, GROUPS, min_n=MIN_N,
                                  n_seeds=N_SEEDS, rows_file=ROWS_FILE)


def plot_on_axes(axes, fold_curves, all_rows_by_grp):
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
            n_variants  = len(all_rows_by_grp[cls][grp])
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
    fold_curves, all_rows_by_grp = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_rows_by_grp)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "protein_class_auroc_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
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
