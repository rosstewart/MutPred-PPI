#!/usr/bin/env python
"""pLDDT quality stratification for Reviewer 1.

Bins test variants by the confidence of the structure AROUND THE MUTATION, then
computes AUROC separately for each bin across 30 seeds × 10 folds = up to 300
per-fold ROC curves. By default that is the mean pLDDT over the mutation's
two-hop contact neighbourhood, matching the subgraph the model is given;
`--scope complex` reproduces the earlier complex-wide mean instead.

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

THE BINS ALSO DIFFER IN COMPOSITION. Low local confidence and intrinsic
disorder are close to the same thing at this granularity: 80% of the low bin's
variants sit in a disordered residue, against 12% of the high bin's.
`disordered_rate` is reported per bin beside `positive_rate` so that is visible
rather than implicit.

Disorder comes from AIUPred (see build_disorder_cache.py), not from AF3's own
`fraction_disordered`, which is one scalar per complex and covers 66% of rows
where AIUPred covers all of them and resolves the mutated residue itself.

Output:
  results/robustness/plddt_auroc_by_class.png
  results/robustness/plddt_auroc_summary.tsv
      columns: class, plddt_bin, mean_auroc, sem, n_variants, n_fold_curves,
               positive_rate, disordered_rate

Usage:
  conda run -n ppi python src/analysis/plddt_stratification.py
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
from analysis import plot_style
from analysis.plot_style import CLASS_LABELS, SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, DATA_ROOT, GCV_RESULTS_DIR, REPO_ROOT, ROBUSTNESS_DIR, cv_reference_dir  # noqa: E402
from analysis.build_plddt_cache import lookup  # pair-keyed cache accessor (either chain order)
from utils.legacy_guard import DATASET_SUFFIX  # noqa: E402
from analysis.stratification_common import (  # noqa: E402
    composition, load_canonical_rows, row_perturbation_labels,
    stratified_fold_curves)


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
# Pair-keyed AF3 cache.  Overridable so a scratch rebuild can be evaluated
# without writing into datasets/annotations/.
PLDDT_CACHE = os.environ.get("MUTPRED_PLDDT_CACHE",
                             str(ANNOTATIONS_DIR / "plddt_pair_cache.pkl"))
GCV_RESULTS = (f"{GCV_RESULTS_DIR}/"
               f"MutPredPPI_sahni_fragoza{DATASET_SUFFIX}_megascale_all"
               f"_detailed_results.pkl")
CANONICAL_DATASET = f"sahni_fragoza{DATASET_SUFFIX}"
CANONICAL_ROWS_PATH = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR = str(ROBUSTNESS_DIR)
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

DISORDER_CACHE = str(ANNOTATIONS_DIR / "disorder_cache.pkl")
RESIDUE_CACHE = str(ANNOTATIONS_DIR / "residue_cache.pkl")

#: Confidence scope. `local` averages pLDDT over the mutation site's `HOPS`-hop
#: contact neighbourhood; `complex` averages over the whole AlphaFold 3 model.
#:
#: LOCAL IS THE DEFAULT BECAUSE IT IS WHAT THE MODEL READS. MutPred-PPI consumes
#: the two-hop subgraph around the mutation (`utils.mutpred_ppi_data`,
#: `two_hop=True`), so a complex-wide mean scores confidence over residues the
#: prediction never touches. The two are only weakly related, Spearman 0.50, and
#: they disagree about the answer: under the complex mean the low-confidence bin
#: has the HIGHEST AUC, which inverts once confidence is measured where the model
#: looks. The complex mean is also confounded with complex size, since low-mean
#: complexes are about 55% longer, and it bins 62/28/10 where the local measure
#: bins 36/31/34.
SCOPE_CHOICES = ("local", "complex")
SCOPE = "local"
HOPS = 2


def _interactor_sequences(dataset: str = CANONICAL_DATASET) -> dict:
    """`'{interactor}-{partner}'` -> interactor sequence.

    Built from the canonical table's own columns and matched whole, for the same
    reason `complex_id_pairs()` is: 261 of the complex_ids contain more than one
    `-`, so splitting the string mis-assigns both proteins for every one.
    """
    from utils.gcv_common import dataset_config, load_data
    df = load_data(dataset_config(dataset))
    return {f"{i}-{p}": str(a) for i, p, a in
            zip(df["interactor"], df["partner"], df["interactor_sequence"])}


def build_row_local_plddt(canonical_rows, hops: int = HOPS,
                          cache_path: str = RESIDUE_CACHE) -> np.ndarray:
    """Mean pLDDT over the mutation site's `hops`-hop contact neighbourhood.

    The neighbourhood spans BOTH chains: a site's contacts include cross-chain
    edges, which is the point -- the confidence of the partner surface it sits
    against is part of what the model reads. Residue pLDDT comes from the
    residue cache (see build_residue_cache.py) and the adjacency from the same
    contact-graph store the model is built on, so this is the input's own
    confidence rather than a re-derivation of it.
    """
    from analysis.build_residue_cache import lookup as residue_lookup
    from analysis.interface_analysis import (TRAIN_EVAL_STORE,
                                             _complex_id_sequences, _mutation_pos1)
    from contact_graphs import ContactGraphStore

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    seqs = _complex_id_sequences()
    store = ContactGraphStore(TRAIN_EVAL_STORE)
    per_pair: dict = {}
    out = np.full(len(canonical_rows), np.nan)
    try:
        for i, row in enumerate(canonical_rows.itertuples(index=False)):
            key = f"{row.interactor}-{row.partner}"
            pair_seqs = seqs.get(key)
            record = residue_lookup(cache, str(row.interactor), str(row.partner))
            if pair_seqs is None or record is None:
                continue
            if key not in per_pair:
                G = store.load_dense(interactor=pair_seqs[0], partner=pair_seqs[1])
                mine = record.get(str(row.interactor))
                theirs = record.get(str(row.partner))
                if G is None or mine is None or theirs is None:
                    per_pair[key] = None
                else:
                    plddt = np.concatenate([mine["plddt"], theirs["plddt"]])
                    n = min(len(plddt), G.shape[0])
                    per_pair[key] = ((G[:n, :n] > 0), plddt[:n])
            entry = per_pair[key]
            if entry is None:
                continue
            adj, plddt = entry
            try:
                pos0 = _mutation_pos1(str(row.mutation)) - 1
            except ValueError:
                continue
            if not 0 <= pos0 < len(plddt):
                continue
            reach = adj[pos0].copy()
            reach[pos0] = True
            for _ in range(hops - 1):
                reach = reach | adj[reach].any(axis=0)
            out[i] = float(plddt[reach].mean())
    finally:
        store.close()
    return out


def build_row_disorder_state(canonical_rows) -> np.ndarray:
    """`'disordered'` / `'structured'` / None per canonical row.

    The state of the MUTATED RESIDUE, from AIUPred at its own calibrated cutoff
    (see build_disorder_cache.py). Rows whose sequence or position cannot be
    resolved stay None and are therefore never selected.
    """
    from analysis.build_disorder_cache import DISORDER_CUTOFF, residue_disorder

    if not os.path.exists(DISORDER_CACHE):
        print(f"  WARNING: {DISORDER_CACHE} not found; structured-only curves "
              f"will be empty (run build_disorder_cache.py)", flush=True)
        return np.array([None] * len(canonical_rows), dtype=object)

    with open(DISORDER_CACHE, "rb") as f:
        cache = pickle.load(f)
    seqs = _interactor_sequences()
    out = []
    for row in canonical_rows.itertuples(index=False):
        seq = seqs.get(f"{row.interactor}-{row.partner}")
        score = None
        if seq is not None:
            m = re.search(r"(\d+)", str(row.mutation))
            if m:
                score = residue_disorder(cache, seq, int(m.group(1)))
        out.append(None if score is None else
                   ("disordered" if score > DISORDER_CUTOFF else "structured"))
    return np.array(out, dtype=object)


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

def compute_curves(scope: str = None):
    """Per-(test class, pLDDT bin) ROC fold curves.

    The reconstruction lives in `analysis.stratification_common`, shared with
    the protein-class and interface supplements; this only builds the binning.
    Row identity is `row_index`, a positional integer into the canonical table
    -- and `load_canonical_rows` now VERIFIES that equivalence rather than
    asserting it in a comment, which is what this script used to do.
    """
    scope = scope or SCOPE
    if scope not in SCOPE_CHOICES:
        raise ValueError(f"unknown scope {scope!r}; expected one of {SCOPE_CHOICES}")
    canonical_rows = load_canonical_rows(CANONICAL_ROWS_PATH)
    print(f"Canonical rows: {len(canonical_rows)}")

    if scope == "local":
        print(f"Computing {HOPS}-hop neighbourhood pLDDT...", flush=True)
        values = build_row_local_plddt(canonical_rows)
        row_plddt_bins = [None if not np.isfinite(v) else bin_plddt(float(v))
                          for v in values]
    else:
        print("Loading pLDDT cache (complex means)...")
        with open(PLDDT_CACHE, "rb") as f:
            plddt_cache = pickle.load(f)
        print(f"  {len(plddt_cache)} AF3 complexes in cache")
        row_plddt_bins = build_row_plddt_bins(canonical_rows, plddt_cache)

    bin_names = [b for b, *_ in PLDDT_BINS]
    bin_counts = {b: sum(1 for v in row_plddt_bins if v == b) for b in bin_names}
    none_count = sum(1 for v in row_plddt_bins if v is None)
    print(f"  pLDDT bins: {bin_counts} | missing: {none_count}")

    return stratified_fold_curves(row_plddt_bins, bin_names, min_n=MIN_N,
                                  n_seeds=N_SEEDS, rows_file=CANONICAL_ROWS_PATH)


SUMMARY_COLUMNS = (
    "class",
    "plddt_bin",
    "mean_auroc",
    "sem",
    "n_variants",
    "n_fold_curves",
    "positive_rate",
    "disordered_rate",
)


def plot_on_axes(axes, fold_curves, all_rows_bin, show_titles: bool = True):
    """Draw the 3-panel (C1/C2/C3) pLDDT-bin ROC comparison onto pre-supplied axes.

    Returns summary_rows (list[str]) for the TSV output.
    """
    bin_names = [b for b, *_ in PLDDT_BINS]
    class_labels = CLASS_LABELS

    labels = row_perturbation_labels()
    disorder_state = build_row_disorder_state(load_canonical_rows(CANONICAL_ROWS_PATH))

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        # Compute all curves first, then sort legend by AUC descending
        plot_entries = []
        for bin_name in bin_names:
            curves = fold_curves[cls][bin_name]
            n_fold_curves = len(curves)
            n_variants, pos_rate = composition(all_rows_bin[cls][bin_name], labels)
            idx = sorted(all_rows_bin[cls][bin_name])
            # Fraction of the stratum whose mutated residue is disordered. Zero
            # by construction for the structured-only subsets, which is the
            # cheapest check that the second pass selected what it meant to.
            sel = disorder_state[np.asarray(idx)] if idx else np.array([], dtype=object)
            known = sel[sel != None]  # noqa: E711 -- object array, `is not None` does not vectorise
            frac_dis = (float((known == "disordered").mean()) if len(known)
                        else float("nan"))
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
                plot_entries.append((mean_auc, bin_name, color, mean_tpr,
                                     lo_tpr, hi_tpr, sem_auc, n_variants,
                                     n_fold_curves))
                summary_rows.append(
                    f"C{cls}\t{bin_name}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}\t{n_fold_curves}\t{pos_rate:.4f}\t{frac_dis:.4f}"
                )
            else:
                summary_rows.append(f"C{cls}\t{bin_name}\tnan\tnan\t{n_variants}\t0\t{pos_rate:.4f}\t{frac_dis:.4f}")

        # Sort descending by AUC before plotting (so legend is AUC-sorted)
        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for (mean_auc, bin_name, color, mean_tpr, lo_tpr, hi_tpr, sem_auc,
             n_variants, _) in plot_entries:
            label = (f"{BIN_LABELS[bin_name]}\n"
                     f"(n={n_variants:,}, AUC={mean_auc:.3f}±{sem_auc:.3f})")
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
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", choices=SCOPE_CHOICES, default=SCOPE,
                   help=f"confidence scope (default: {SCOPE})")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    fold_curves, all_rows_bin = compute_curves(args.scope)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_rows_bin)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "plddt_auroc_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_tsv = os.path.join(OUT_DIR, "plddt_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
