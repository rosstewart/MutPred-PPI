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
import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import DATA_ROOT, REPO_ROOT, cv_reference_dir
from variant_db_inference import variant_rows as vr
from analysis.stratification_common import (  # noqa: E402
    load_canonical_rows, stratified_fold_curves)


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
from analysis.gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)


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
    from utils.gcv_common import dataset_config, load_data
    df = load_data(dataset_config(dataset))
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
    """Per-(test class, interface-vs-non-interface) ROC fold curves.

    The reconstruction lives in `analysis.stratification_common`, shared with
    the protein-class and pLDDT supplements; this only builds the flags.
    """
    canonical_rows = load_canonical_rows(ROWS_FILE)
    print(f"Canonical rows: {len(canonical_rows)}")

    flags = build_row_iface_flags(canonical_rows)       # list[bool|None]
    # The plotting code and the colour/label tables key on these strings, so
    # translate once here rather than carrying booleans through and mapping at
    # every use. None (no structure / position not resolvable) stays None and is
    # therefore never selected.
    row_groups = [None if v is None else ("interface" if v else "non_interface")
                  for v in flags]
    groups = ["interface", "non_interface"]
    counts = {g: row_groups.count(g) for g in groups}
    print(f"  interface flags: {counts} | missing: {row_groups.count(None)}")

    return stratified_fold_curves(row_groups, groups, min_n=MIN_N,
                                  n_seeds=N_SEEDS, rows_file=ROWS_FILE)


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
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "interface_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("class\tgroup\tmean_auroc\tsem\tn_variants\tn_fold_curves\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
