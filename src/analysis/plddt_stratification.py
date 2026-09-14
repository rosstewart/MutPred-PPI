#!/usr/bin/env python
"""pLDDT quality stratification for Reviewer 1.

Bins test complexes by the mean pLDDT of OUR OWN AF3 model of that complex,
then computes AUROC separately for each bin across 30 seeds × 10 folds = up to
300 per-fold ROC curves.

The confidence values come from datasets/annotations/plddt_pair_cache.pkl, a
pair-keyed `{"A__B": float}` map built by src/analysis/build_plddt_cache.py from
the AF3 canonical manifests.  It replaces an accession-keyed cache of AlphaFold
DB *monomer* models, which had to average two separately-folded chains and had
no isoform resolution.  Because the key is the pair, the isoform-stripping
`split("-")[0]` fallback this module used to need is gone.

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10) (hardcoded denominator)

Confidence bins (per complex):
  Low:    mean pLDDT < 70
  Medium: 70 ≤ mean pLDDT < 85
  High:   mean pLDDT ≥ 85

Output:
  results/robustness/plddt_auroc_by_class.png
  results/robustness/plddt_auroc_summary.tsv

Usage:
  conda run -n ppi python src/analysis/plddt_stratification.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
from analysis import plot_style
from analysis.plot_style import CLASS_LABELS, SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, DATA_ROOT, REPO_ROOT, cv_reference_dir
from analysis.build_plddt_cache import lookup  # pair-keyed cache accessor (either chain order)
from utils.legacy_guard import DATASET_SUFFIX  # noqa: E402
from analysis.stratification_common import (  # noqa: E402
    load_canonical_rows, stratified_fold_curves)


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
# Pair-keyed AF3 cache.  Overridable so a scratch rebuild can be evaluated
# without writing into datasets/annotations/.
PLDDT_CACHE = os.environ.get("MUTPRED_PLDDT_CACHE",
                             str(ANNOTATIONS_DIR / "plddt_pair_cache.pkl"))
GCV_RESULTS = (f"{_PUB}/results/gcv/"
               f"MutPredPPI_sahni_fragoza{DATASET_SUFFIX}_megascale_all"
               f"_detailed_results.pkl")
CANONICAL_DATASET = f"sahni_fragoza{DATASET_SUFFIX}"
CANONICAL_ROWS_PATH = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR = f"{_PUB}/results/robustness"
N_SEEDS = 30
MIN_N = 5  # matches roc_plots.py spirit: just require both label classes per fold
from analysis.gcv_curves import FPR_GRID, N_SEM_DIVISOR  # noqa: E402  (single definition)

PLDDT_BINS = [
    ("low",    0,    70),
    ("medium", 70,   85),
    ("high",   85,  100),
]
BIN_COLORS = {"low": "#d6604d", "medium": "#f4a582", "high": "#4393c3"}
BIN_LABELS = {"low": "Low (<70)", "medium": "Medium (70–85)", "high": "High (≥85)"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def complex_id_pairs(dataset: str = f"sahni_fragoza{DATASET_SUFFIX}") -> dict:
    """`'{interactor}-{partner}'` -> (interactor, partner), built by JOIN not split.

    `complex_id` welds two accessions with `-`, but `-` also introduces an isoform
    suffix, so the string is ambiguous: 261 of the 2,785 complex_ids here contain
    more than one `-` and `split("-")` mis-assigns both proteins for every one of
    them (`P60891-1-B4DP31` becomes `('P60891', '1-B4DP31')`). Constructing the
    key from the canonical table's own columns and matching it whole is exact.
    """
    from utils.gcv_common import dataset_config, load_data
    df = load_data(dataset_config(dataset))
    return {f"{i}-{p}": (i, p) for i, p in zip(df["interactor"], df["partner"])}


def _split_against_cache(complex_id: str, plddt_cache: dict):
    """Recover the two proteins of an unjoinable complex_id using the cache itself.

    Only reached for complex_ids absent from the canonical table.  Every `-` is
    tried as the weld point and the candidate is kept only if the resulting pair
    is a complex we actually modelled; a unique hit is the answer, zero or more
    than one is a refusal.  This is exact where the historical `split("-")` was
    a guess: it never invents a protein that has no structure behind it.
    """
    hits = set()
    for i, ch in enumerate(complex_id):
        if ch != "-":
            continue
        a, b = complex_id[:i], complex_id[i + 1:]
        if lookup(plddt_cache, a, b) is not None:
            hits.add((a, b))
    return hits.pop() if len(hits) == 1 else None


def complex_mean_plddt(complex_id: str, plddt_cache: dict, pairs: dict | None = None):
    """Mean pLDDT of the AF3 model of this complex, or None if we have no model.

    One cache lookup, one scalar: the cache is keyed by pair, so there is no
    per-chain averaging and no isoform-stripping fallback here.
    """
    prots = (pairs or {}).get(complex_id) or _split_against_cache(complex_id, plddt_cache)
    if prots is None:
        return None
    val = lookup(plddt_cache, *prots)
    return None if val is None else float(val)


def bin_plddt(val: float) -> str:
    for name, lo, hi in PLDDT_BINS:
        if lo <= val < hi:
            return name
    return "high"


def build_row_plddt_bins(canonical_rows: pd.DataFrame, plddt_cache: dict) -> list:
    """Per-canonical-row pLDDT bin (or None), in `row_index` order.

    Returns a list of length `len(canonical_rows)` where `result[i]` is the
    pLDDT bin of the pair at row `i`. Keying by row position rather than by a
    vt_id string removes the dependency on the pre-090826 vt_ids pickles, which
    had 5,894 rows against the current 6,219-row canonical table.

    All pairs share a pLDDT with every mutation in that pair, so the per-pair
    result is cached (`seen`) and re-used for repeat pairs.
    """
    pairs_map = complex_id_pairs()
    seen: dict[tuple[str, str], str | None] = {}
    result: list[str | None] = []
    unjoined: set[str] = set()
    rescued: set[str] = set()

    for _, row in canonical_rows.iterrows():
        interactor, partner = str(row["interactor"]), str(row["partner"])
        key = (interactor, partner)
        if key not in seen:
            cid = f"{interactor}-{partner}"
            if cid not in pairs_map:
                unjoined.add(cid)
            mean_val = complex_mean_plddt(cid, plddt_cache, pairs_map)
            if mean_val is not None and cid in unjoined:
                rescued.add(cid)
            seen[key] = bin_plddt(mean_val) if mean_val is not None else None
        result.append(seen[key])

    n_missing = sum(1 for v in seen.values() if v is None)
    print(f"  {len(seen)} unique pairs: {len(seen) - n_missing} binned, {n_missing} "
          f"with no AF3 model; {len(unjoined)} not in complex_id_pairs() "
          f"({len(rescued)} of those rescued from the pLDDT cache's pair set)", flush=True)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def compute_curves():
    """Per-(test class, pLDDT bin) ROC fold curves.

    The reconstruction lives in `analysis.stratification_common`, shared with
    the protein-class and interface supplements; this only builds the binning.
    Row identity is `row_index`, a positional integer into the canonical table
    -- and `load_canonical_rows` now VERIFIES that equivalence rather than
    asserting it in a comment, which is what this script used to do.
    """
    canonical_rows = load_canonical_rows(CANONICAL_ROWS_PATH)
    print(f"Canonical rows: {len(canonical_rows)}")

    print("Loading pLDDT cache...")
    with open(PLDDT_CACHE, "rb") as f:
        plddt_cache = pickle.load(f)
    print(f"  {len(plddt_cache)} AF3 complexes in cache")

    row_plddt_bins = build_row_plddt_bins(canonical_rows, plddt_cache)
    bin_names = [b for b, *_ in PLDDT_BINS]
    bin_counts = {b: sum(1 for v in row_plddt_bins if v == b) for b in bin_names}
    none_count = sum(1 for v in row_plddt_bins if v is None)
    print(f"  pLDDT bins: {bin_counts} | missing: {none_count}")

    return stratified_fold_curves(row_plddt_bins, bin_names, min_n=MIN_N,
                                  n_seeds=N_SEEDS,
                                  rows_file=CANONICAL_ROWS_PATH)


def plot_on_axes(axes, fold_curves, all_rows_bin, show_titles: bool = True):
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
            n_variants = len(all_rows_bin[cls][bin_name])
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

        # In the combined figure only the top row is labelled: the C1/C2/C3
        # columns are shared across all three panels, so repeating the
        # titles on every row is noise.
        if show_titles:
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
    fold_curves, all_rows_bin = compute_curves()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_rows_bin)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "plddt_auroc_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "plddt_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("class\tplddt_bin\tmean_auroc\tsem\tn_variants\tn_fold_curves\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
