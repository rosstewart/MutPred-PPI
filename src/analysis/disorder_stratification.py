#!/usr/bin/env python
"""Disordered vs. structured mutation-site stratification.

Splits test variants by whether the WILD-TYPE residue carrying the mutation is
predicted to be intrinsically disordered, then computes AUROC separately per
group across 30 seeds x 10 folds of existing GCV test predictions -- no new GCV
run.

WHY THIS IS ITS OWN PANEL. Disorder is a property of the variant, like the
structural context in the interface panel and the domain architecture in the
protein-class one. It is also the quantity the complex-confidence panel controls
for, so it has to be defined before that panel uses it.

WHAT THE PANEL CAN AND CANNOT SHOW. An ROC panel shows discrimination. The other
half of the disorder result is class balance -- disordered sites are disrupted
far less often than structured ones -- and no ROC curve can display that, so
`positive_rate` is reported per stratum in the summary TSV and is the number to
quote alongside the curves.

Disorder comes from AIUPred at its own calibrated cutoff; see
build_disorder_cache.py for why it is used in preference to AlphaFold 3's
per-complex `fraction_disordered`.

Output:
  results/robustness/disorder_auroc_by_class.png
  results/robustness/disorder_auroc_summary.tsv

Usage:
  conda run -n ppi python src/analysis/disorder_stratification.py
"""

import os

import numpy as np
import matplotlib
from analysis import plot_style
from analysis.plot_style import CLASS_LABELS, SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

from paths import ROBUSTNESS_DIR  # noqa: E402
from analysis.gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402
from analysis.stratification_common import (  # noqa: E402
    MIN_N, N_SEEDS, composition, load_canonical_rows, row_perturbation_labels,
    stratified_fold_curves)
from analysis.plddt_stratification import (  # noqa: E402
    CANONICAL_ROWS_PATH, build_row_disorder_state)

OUT_DIR = str(ROBUSTNESS_DIR)

GROUPS = ["structured", "disordered"]
COLORS = {"structured": "#5e3c99", "disordered": "#b2abd2"}
LABELS = {"structured": "Structured site", "disordered": "Disordered site"}

SUMMARY_COLUMNS = (
    "class",
    "group",
    "mean_auroc",
    "sem",
    "n_variants",
    "n_fold_curves",
    "positive_rate",
)


def compute_curves():
    """Per-(test class, disorder state) ROC fold curves."""
    rows = load_canonical_rows(CANONICAL_ROWS_PATH)
    print(f"Canonical rows: {len(rows)}", flush=True)
    state = build_row_disorder_state(rows)
    counts = {g: int((state == g).sum()) for g in GROUPS}
    print(f"  disorder states: {counts} | missing: {int((state == None).sum())}",  # noqa: E711
          flush=True)
    return stratified_fold_curves(list(state), GROUPS, min_n=MIN_N,
                                  n_seeds=N_SEEDS, rows_file=CANONICAL_ROWS_PATH)


def plot_on_axes(axes, fold_curves, all_rows_group, show_titles: bool = True):
    """Draw the 3-panel (C1/C2/C3) ROC comparison onto pre-supplied axes."""
    labels = row_perturbation_labels()
    summary_rows = []

    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        plot_entries = []
        for key in GROUPS:
            curves = fold_curves[cls][key]
            n_fold_curves = len(curves)
            n_variants, pos_rate = composition(all_rows_group[cls][key], labels)
            if curves:
                arr = np.array(curves)
                mean_tpr = np.mean(arr, axis=0)
                sem_tpr = np.std(arr, axis=0, ddof=1) / np.sqrt(N_SEM_DIVISOR)
                per_fold = [np.trapz(c, FPR_GRID) for c in curves]
                mean_auc = float(np.mean(per_fold))
                sem_auc = float(np.std(per_fold, ddof=1) / np.sqrt(N_SEM_DIVISOR))
                plot_entries.append((mean_auc, key, mean_tpr, mean_tpr - sem_tpr,
                                     mean_tpr + sem_tpr, sem_auc, n_variants))
                summary_rows.append(
                    f"C{cls}\t{key}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}"
                    f"\t{n_fold_curves}\t{pos_rate:.4f}")
            else:
                summary_rows.append(
                    f"C{cls}\t{key}\tnan\tnan\t{n_variants}\t0\t{pos_rate:.4f}")

        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, key, mean_tpr, lo, hi, sem_auc, n_variants in plot_entries:
            ax.plot(FPR_GRID, mean_tpr, color=COLORS[key], lw=2,
                    label=f"{LABELS[key]}\n(n={n_variants:,}, "
                          f"AUC={mean_auc:.3f}±{sem_auc:.3f})")
            ax.fill_between(FPR_GRID, lo, hi, color=COLORS[key], alpha=0.15)

        if show_titles:
            ax.set_title(CLASS_LABELS[cls], fontsize=10)
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
    out_png = os.path.join(OUT_DIR, "disorder_auroc_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "disorder_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
