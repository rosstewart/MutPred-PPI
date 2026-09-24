#!/usr/bin/env python
"""Single-domain vs multi-domain stratification for Reviewer 1.

Classifies each test interactor (the protein carrying the mutation) as
single-domain or multi-domain, then computes AUROC separately per group across
30 seeds × 10 folds of existing GCV test predictions — no new GCV run.

HOW A DOMAIN IS COUNTED (`--domain-source`):

  ipr_all      len(set(InterPro accessions)) from `pfam_domains_cache.pkl`.
               What the published figure used. It counts every InterPro entry
               TYPE together -- Family, Domain, Conserved_site,
               Homologous_superfamily -- so a genuinely single-domain protein
               routinely accrues three to five accessions and `n == 1` almost
               never fires: only 125 of 909 interactors (13.8%) come out
               single-domain.
  ipr_domain   InterPro restricted to `?type=domain`, dropping the family and
               superfamily entries the rule above counted.
  ted          TED (https://ted.cathdb.info), a structure-based domain parse of
               the AlphaFold models, which gives an unambiguous count plus
               residue ranges.

`ipr_domain` and `ted` are read from `results/protein_class/domain_annotations.csv`,
built by `fetch_ted_domains.py`, which prints the three sources side by side.
`ted` is the default. The comparison that settled it, over the 968 interactors:

    source        coverage   single-domain   C1 single/multi   C2          C3
    ipr_all       909         13.8%          0.625 / 0.859     0.78/0.83   0.899 / 0.750
    ipr_domain    968         27.1%          0.870 / 0.859     0.84/0.81   0.765 / 0.737
    ted           968         36.5%          0.835 / 0.861     0.76/0.84   0.701 / 0.764

(`ipr_domain`'s AUCs are over the 734 proteins it covered before HTTP 204 was
recognised as "no entry of this type"; its coverage is now 968 as well, and the
234 it gained have zero Domain-type entries so they stay unlabelled.)

Under `ipr_all` the two extreme numbers rest on 250 and 54 of 300 fold curves;
under `ted` no stratum falls below 287. TED and InterPro agree on the
single-domain RATE to within 10 points (36.5% vs 27.1%) while disagreeing on 31%
of individual
proteins, so the rate is the robust part and neither is treated as ground truth.

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
from analysis.plot_style import CLASS_LABELS, SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, DATA_ROOT, GCV_RESULTS_DIR, PROTEIN_CLASS_DIR, REPO_ROOT, ROBUSTNESS_DIR, cv_reference_dir  # noqa: E402
from utils.legacy_guard import DATASET_SUFFIX  # noqa: E402
from analysis.stratification_common import (  # noqa: E402
    composition, load_canonical_rows, row_perturbation_labels,
    stratified_fold_curves)
from utils.identifiers import bare_accession  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
_BASE = str(DATA_ROOT)
CV_DIR = str(cv_reference_dir())
PFAM_CACHE = str(ANNOTATIONS_DIR / "pfam_domains_cache.pkl")
GCV_RESULTS = (f"{GCV_RESULTS_DIR}/"
               f"MutPredPPI_sahni_fragoza{DATASET_SUFFIX}_megascale_all"
               f"_detailed_results.pkl")
CANONICAL_DATASET = f"sahni_fragoza{DATASET_SUFFIX}"
# Canonical row ordering: row_index indexes the fold splits and test classes.
ROWS_FILE = f"{CV_DIR}/sahni_fragoza_train_rows.csv.gz"
OUT_DIR   = str(ROBUSTNESS_DIR)

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

DOMAIN_SOURCES = ("ipr_all", "ipr_domain", "ted")
DOMAIN_TABLE = str(PROTEIN_CLASS_DIR / "domain_annotations.csv")

#: Default source. TED, chosen from the three-way comparison `fetch_ted_domains.py`
#: prints: it and InterPro both cover all 968 interactors, but TED is the only one
#: whose direction is consistent across C1/C2/C3,
#: and the only one where every stratum keeps 287-300 of its 300 fold curves.
#: Under `ipr_all` the C3 single-domain curve rested on 54.
DOMAIN_SOURCE = "ted"


def build_domain_lookup(pfam_hits: dict) -> dict:
    """UniProt ID → 'single' | 'multi' | None."""
    out = {}
    for uid, ipr_list in pfam_hits.items():
        n = len(set(ipr_list))
        out[uid] = "single" if n == 1 else "multi"
    return out


def load_domain_lookup(source: str = DOMAIN_SOURCE) -> dict:
    """UniProt ID → 'single' | 'multi', from whichever domain definition is asked for.

    A protein the source has no answer for is absent from the dict rather than
    labelled, so it falls through to "unknown" and is never selected -- the same
    treatment the InterPro path has always given a protein with no hits.
    """
    if source not in DOMAIN_SOURCES:
        raise ValueError(f"unknown domain source {source!r}; expected one of "
                         f"{DOMAIN_SOURCES}")
    if source == "ipr_all":
        with open(PFAM_CACHE, "rb") as f:
            return build_domain_lookup(pickle.load(f)["hits"])

    if not os.path.exists(DOMAIN_TABLE):
        raise FileNotFoundError(
            f"{DOMAIN_TABLE} not found; run "
            f"src/analysis/fetch_ted_domains.py to build it.")
    df = pd.read_csv(DOMAIN_TABLE)
    column = {"ipr_domain": "n_ipr_domain", "ted": "n_ted"}[source]
    counts = df.dropna(subset=[column])
    return {str(u): ("single" if int(n) == 1 else "multi")
            for u, n in zip(counts["uniprot_id"], counts[column]) if int(n) >= 1}


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

def compute_curves(domain_source: str = None):
    """Per-(test class, domain-group) ROC fold curves.

    The reconstruction itself lives in `analysis.stratification_common`, shared
    with the pLDDT and interface supplements; this only builds the grouping.
    """
    source = domain_source or DOMAIN_SOURCE
    print(f"Loading domain annotations (source: {source})...", flush=True)
    domain_lookup = load_domain_lookup(source)
    print(f"  {len(domain_lookup)} proteins with a domain count", flush=True)

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


def plot_on_axes(axes, fold_curves, all_rows_by_grp, show_titles: bool = True):
    """Draw the 3-panel (C1/C2/C3) single-vs-multi-domain ROC comparison onto
    pre-supplied axes. Returns summary_rows (list[str]) for the TSV output.
    """
    class_labels = CLASS_LABELS

    labels = row_perturbation_labels()

    summary_rows = []
    for ax, cls in zip(axes, (1, 2, 3)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        plot_entries = []

        for grp in GROUPS:
            curves      = fold_curves[cls][grp]
            n_fold      = len(curves)
            n_variants, pos_rate = composition(all_rows_by_grp[cls][grp], labels)
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
                    f"C{cls}\t{grp}\t{mean_auc:.4f}\t{sem_auc:.4f}\t{n_variants}\t{n_fold}\t{pos_rate:.4f}"
                )
            else:
                summary_rows.append(f"C{cls}\t{grp}\tnan\tnan\t{n_variants}\t0\t{pos_rate:.4f}")

        plot_entries.sort(reverse=True, key=lambda e: e[0])
        for mean_auc, grp, color, mean_tpr, lo_tpr, hi_tpr, sem_auc, n_variants, _ in plot_entries:
            label = (f"{LABELS[grp]}\n"
                     f"(n={n_variants:,}, AUC={mean_auc:.3f}±{sem_auc:.3f})")
            ax.plot(FPR_GRID, mean_tpr, color=color, lw=2, label=label)
            ax.fill_between(FPR_GRID, lo_tpr, hi_tpr, color=color, alpha=0.15)

        # In the combined figure only the top row is labelled: the C1/C2/C3
        # columns are shared across all three panels, so repeating the
        # titles on every row is noise.
        if show_titles:
            ax.set_title(class_labels[cls], fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
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
    p.add_argument("--domain-source", choices=DOMAIN_SOURCES, default=DOMAIN_SOURCE,
                   help=f"how a domain is counted (default: {DOMAIN_SOURCE})")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="where to write the panel and summary (default: %(default)s)")
    args = p.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    fold_curves, all_rows_by_grp = compute_curves(args.domain_source)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    summary_rows = plot_on_axes(axes, fold_curves, all_rows_by_grp)

    plt.tight_layout()
    out_png = os.path.join(out_dir, "protein_class_auroc_by_class.png")
    plt.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved: {out_png}", flush=True)

    out_tsv = os.path.join(out_dir, "protein_class_auroc_summary.tsv")
    with open(out_tsv, "w") as f:
        f.write("\t".join(SUMMARY_COLUMNS) + "\n")
        f.write("\n".join(summary_rows) + "\n")
    print(f"Saved: {out_tsv}", flush=True)

    # No symlink into figures/. This panel is an intermediate: the manuscript
    # figure is the combined three-row robustness figure that
    # combined_robustness_figure.py assembles from this and its two siblings.
    # figures/ is rebuilt from the notebook's manifest, so a link planted here
    # would be pruned on the next run anyway, and neither sibling plants one.


if __name__ == "__main__":
    main()
