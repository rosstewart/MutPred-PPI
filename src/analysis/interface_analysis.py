#!/usr/bin/env python
"""Interface vs. surface vs. buried variant analysis for Reviewer 1.

For each of 30 GCV seeds × 10 folds = up to 300 per-fold ROC curves,
reconstructs where in the structure each test variant sits, then computes AUROC
separately per group across the three test classes (C1/C2/C3).

THE COMPARATOR IS SPLIT IN TWO. The interface rule is unchanged -- the mutated
residue makes at least one cross-chain contact in the AF3 contact graph -- but
its complement is no longer reported as one group. "Non-interface" held the
buried core together with the free surface, and 1,623 of its 5,612 variants
(29%) are buried, so a reader comparing against it was comparing against a
mixture rather than against the surface control they had in mind. The three
groups are now:

    interface                 >=1 cross-chain contact
    non-interface, surface    no cross-chain contact, RSA >= 0.25
    non-interface, buried     no cross-chain contact, RSA < 0.25

Both comparator groups are defined WITHIN the non-interface set, and their names
say so. An interface residue is usually solvent-exposed itself (82% sit at
RSA >= 0.25), so "surface" on its own would be read as including the interface;
and 18% of interface residues are buried by RSA, so "buried" on its own is
equally partial.

RSA is unbound relative solvent accessibility from
`datasets/annotations/residue_cache.pkl` (see build_residue_cache.py), computed
on the ISOLATED chain so that a residue is not called buried merely because its
partner covers it, and referenced to the Tien et al. 2013 theoretical maxima.

A fourth curve, `interface_hc`, is the interface restricted to contacts that are
confident on BOTH sides: the mutated residue's own pLDDT >= 70 and the mean
pLDDT of the partner residues it touches >= 70. The partner-side condition
removes contacts made by a disordered stretch of the partner wrapping onto a
well-modelled site. It is a SUBSET of `interface`, so it is computed in its own
pass: `stratified_fold_curves` assigns one group per row and cannot take
overlapping groups.

Aggregation matches compute_roc_with_variance() in roc_plots.py exactly:
  - 100 FPR interpolation points
  - Mean ± SEM where SEM = std / sqrt(10) (hardcoded denominator)

Output:
  results/robustness/interface_auroc_by_class.png
  results/robustness/interface_auroc_summary.tsv
      columns: class, group, mean_auroc, sem, n_variants, n_fold_curves,
               positive_rate -- the fraction of the stratum whose interaction is
               disrupted. Strata differ in class balance, so an AUC gap is only
               a skill gap once the positive rates are comparable.

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
from analysis.plot_style import CLASS_LABELS, SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import DATA_ROOT, GCV_RESULTS_DIR, REPO_ROOT, ROBUSTNESS_DIR, contact_graph_store, cv_reference_dir  # noqa: E402
from variant_db_inference import variant_rows as vr
from utils.legacy_guard import DATASET_SUFFIX  # noqa: E402
from analysis.stratification_common import (  # noqa: E402
    ROWS_FILE, composition, load_canonical_rows, row_perturbation_labels,
    stratified_fold_curves)
from analysis.build_residue_cache import lookup as residue_lookup  # noqa: E402
from paths import ANNOTATIONS_DIR as _ANNOTATIONS_DIR  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
TRAIN_EVAL_STORE = str(contact_graph_store())
GCV_RESULTS = (f"{GCV_RESULTS_DIR}/"
               f"MutPredPPI_sahni_fragoza{DATASET_SUFFIX}_megascale_all"
               f"_detailed_results.pkl")
CANONICAL_DATASET = f"sahni_fragoza{DATASET_SUFFIX}"
CANONICAL_ROWS_PATH = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR = str(ROBUSTNESS_DIR)
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


def _complex_id_sequences(dataset: str = f"sahni_fragoza{DATASET_SUFFIX}") -> dict:
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

RESIDUE_CACHE = str(_ANNOTATIONS_DIR / "residue_cache.pkl")

#: Relative solvent accessibility above which a non-interface residue counts as
#: surface rather than buried. 25% is the long-standing convention (Rost and
#: Sander 1994); it is a convention, not a measurement, so it is named here and
#: stated in Methods alongside the Tien et al. 2013 reference maxima that RSA
#: is computed against.
RSA_SURFACE = 0.25

#: Per-residue pLDDT above which an interface assignment is treated as
#: confident, applied to BOTH sides of the contact. AlphaFold's own "confident"
#: band starts at 70. Raising it to 85 was tested and is worse on every axis:
#: it keeps 206 of 600 interface variants, and one class-two cell falls to 55
#: variants on 123 of 300 fold curves.
PLDDT_CONFIDENT = 70.0

#: Disjoint, and reported as the figure's three curves.
GROUPS = ["interface", "surface", "buried"]

#: A subset of `interface`, so it needs its own pass.
CONFIDENCE_GROUPS = ["interface_hc"]


def build_row_residue_properties(canonical_rows: pd.DataFrame,
                                 cache_path: str = RESIDUE_CACHE):
    """`(rsa, plddt)` per canonical row, in `row_index` order, NaN where unknown.

    The residue is located in the INTERACTOR chain, keyed by accession within
    the pair record -- not by chain letter. AF3 chain order is unrelated to the
    (interactor, partner) order of the row table, so a chain-letter lookup would
    silently read the partner's residue for roughly half the pairs.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    rsa = np.full(len(canonical_rows), np.nan)
    plddt = np.full(len(canonical_rows), np.nan)
    no_pair = no_chain = out_of_range = 0

    for i, row in enumerate(canonical_rows.itertuples(index=False)):
        record = residue_lookup(cache, str(row.interactor), str(row.partner))
        if record is None:
            no_pair += 1
            continue
        chain = record.get(str(row.interactor))
        if chain is None:
            no_chain += 1
            continue
        try:
            pos0 = _mutation_pos1(str(row.mutation)) - 1
        except ValueError:
            continue
        if not 0 <= pos0 < len(chain["rsa"]):
            out_of_range += 1
            continue
        rsa[i] = chain["rsa"][pos0]
        plddt[i] = chain["plddt"][pos0]

    for n, what in ((no_pair, "pair absent from the residue cache"),
                    (no_chain, "interactor accession absent from the pair record"),
                    (out_of_range, "mutation position outside the modelled chain")):
        if n:
            print(f"  {n} rows: {what}")
    return rsa, plddt


def build_row_interface_plddt(canonical_rows: pd.DataFrame, cache_path: str = RESIDUE_CACHE,
                              store=None):
    """`(site_plddt, partner_contact_plddt)` per canonical row, NaN where unknown.

    The second array is the mean pLDDT over the PARTNER residues the mutated
    residue actually touches, taken from that residue's cross-chain edges in the
    contact graph. Thresholding the site alone leaves in contacts made by a
    disordered stretch of the partner that wraps onto a well-modelled site, which
    is a confident-looking interface assignment resting on an unconfident
    partner. The two are only weakly related (Spearman 0.33 over the 600
    interface variants), so this is not a restatement of the site's own pLDDT.
    """
    from contact_graphs import ContactGraphStore

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    seq_map = _complex_id_sequences()
    own_store = store is None
    if own_store:
        store = ContactGraphStore(TRAIN_EVAL_STORE)

    site = np.full(len(canonical_rows), np.nan)
    partner = np.full(len(canonical_rows), np.nan)
    try:
        for i, row in enumerate(canonical_rows.itertuples(index=False)):
            seqs = seq_map.get(f"{row.interactor}-{row.partner}")
            if seqs is None:
                continue
            try:
                pos0 = _mutation_pos1(str(row.mutation)) - 1
            except ValueError:
                continue
            if not 0 <= pos0 < len(seqs[0]):
                continue
            record = residue_lookup(cache, str(row.interactor), str(row.partner))
            if record is None:
                continue
            mine = record.get(str(row.interactor))
            theirs = record.get(str(row.partner))
            if mine is None or pos0 >= len(mine["plddt"]):
                continue
            site[i] = mine["plddt"][pos0]
            G = store.load_dense(interactor=seqs[0], partner=seqs[1])
            if G is None or theirs is None:
                continue
            contacts = np.nonzero(G[pos0, len(seqs[0]):] > 0)[0]
            contacts = contacts[contacts < len(theirs["plddt"])]
            if len(contacts):
                partner[i] = float(np.mean(theirs["plddt"][contacts]))
    finally:
        if own_store:
            store.close()
    return site, partner


def compute_curves():
    """Per-(test class, structural group) ROC fold curves.

    The reconstruction lives in `analysis.stratification_common`, shared with
    the protein-class and pLDDT supplements; this only builds the groupings.

    Two passes, because `interface_hc` is a subset of `interface` and
    `stratified_fold_curves` takes one group per row.
    """
    canonical_rows = load_canonical_rows(ROWS_FILE)
    print(f"Canonical rows: {len(canonical_rows)}")

    flags = build_row_iface_flags(canonical_rows)       # list[bool|None]
    rsa, plddt = build_row_residue_properties(canonical_rows)

    # The plotting code and the colour/label tables key on these strings, so
    # translate once here rather than carrying booleans through and mapping at
    # every use. None (no structure / position not resolvable) stays None and is
    # therefore never selected.
    row_groups = []
    for flag, r in zip(flags, rsa):
        if flag is None:
            row_groups.append(None)
        elif flag:
            row_groups.append("interface")
        elif np.isnan(r):
            row_groups.append(None)
        else:
            row_groups.append("surface" if r >= RSA_SURFACE else "buried")

    counts = {g: row_groups.count(g) for g in GROUPS}
    print(f"  structural groups: {counts} | missing: {row_groups.count(None)}")

    site_plddt, partner_plddt = build_row_interface_plddt(canonical_rows)
    hc_groups = ["interface_hc"
                 if (g == "interface" and sp >= PLDDT_CONFIDENT
                     and pp >= PLDDT_CONFIDENT)
                 else None
                 for g, sp, pp in zip(row_groups, site_plddt, partner_plddt)]
    print(f"  interface_hc (site and partner-contact pLDDT >= "
          f"{PLDDT_CONFIDENT:.0f}): {hc_groups.count('interface_hc')}")

    fold_curves, all_rows = stratified_fold_curves(
        row_groups, GROUPS, min_n=MIN_N, n_seeds=N_SEEDS, rows_file=ROWS_FILE)
    hc_curves, hc_rows = stratified_fold_curves(
        hc_groups, CONFIDENCE_GROUPS, min_n=MIN_N, n_seeds=N_SEEDS,
        rows_file=ROWS_FILE)

    for cls in fold_curves:
        fold_curves[cls].update(hc_curves[cls])
        all_rows[cls].update(hc_rows[cls])
    return fold_curves, all_rows


# Restored: centralising the shared palette into plot_style dropped these,
# which are specific to this figure, not shared. Interface and its
# high-confidence subset share a hue so the nesting reads off the legend.
CURVE_COLORS = {"interface": "#1f77b4", "interface_hc": "#08306b",
                "surface": "#aec7e8", "buried": "#6b6b6b"}
# Both non-interface groups say so. Interface residues are themselves mostly
# solvent-exposed (82% at RSA >= 25%), so a bare "Surface" reads as including
# them, which is the confusion this three-way split exists to remove; and 18% of
# interface residues are buried by RSA, so a bare "Buried" is equally partial.
DISPLAY_NAMES = {"interface": "Interface", "interface_hc": "Interface (confident, pLDDT \u2265 70)",
                 "surface": "Non-interface, surface",
                 "buried": "Non-interface, buried"}


#: Column order of the summary TSV, shared with
#: `export_reconstruction_tables._export_one_robustness` so the exported
#: CSV cannot drift from the TSV when a column is added.
SUMMARY_COLUMNS = (
    "class",
    "group",
    "mean_auroc",
    "sem",
    "n_variants",
    "n_fold_curves",
    "positive_rate",
)


def plot_on_axes(axes, fold_curves, all_rows_group, show_titles: bool = True):
    """Draw the 3-panel (C1/C2/C3) ROC comparison onto pre-supplied axes.

    Returns summary_rows (list[str]) for the TSV output.
    """
    class_labels = CLASS_LABELS
    curve_colors = CURVE_COLORS
    display_names = DISPLAY_NAMES
    labels = row_perturbation_labels()

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

        # Compute all curves first, then sort legend by AUC descending
        plot_entries = []
        for key in GROUPS + CONFIDENCE_GROUPS:
            curves = fold_curves[cls][key]
            n_fold_curves = len(curves)
            n_variants, pos_rate = composition(all_rows_group[cls][key], labels)
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
                    f"C{cls}\t{key}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}"
                    f"\t{n_fold_curves}\t{pos_rate:.4f}"
                )
            else:
                summary_rows.append(
                    f"C{cls}\t{key}\tnan\tnan\t{n_variants}\t0\t{pos_rate:.4f}")

        # Sort descending by AUC before plotting (so legend is AUC-sorted)
        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, key, color, mean_tpr, lo_tpr, hi_tpr, sem_auc, n_variants, _ in plot_entries:
            label = (f"{display_names[key]}\n"
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
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()
