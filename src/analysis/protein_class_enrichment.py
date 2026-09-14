#!/usr/bin/env python3
"""Protein class enrichment for ClinVar pathogenic vs all-gnomAD baseline.

For each broad GO-based functional class of interacting protein, computes
Quasi-null and Edgetic enrichment within ClinVar pathogenic variants vs the
all-gnomAD baseline.  Uses the identical multinomial bootstrap, calc_enrichment,
and draw_component logic as variant_db_charts.py so the figure matches Fig 5.

Classes are derived from GO annotations fetched from UniProt REST API.

Usage:
    conda run -n ppi python src/analysis/protein_class_enrichment.py

Output:
    results/protein_class/pathogenic_by_class.png
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
from analysis import edgotypes  # noqa: E402
from analysis import training_overlap  # noqa: E402
from analysis.classify_variant_dbs import (  # noqa: E402
    group_by_variant, load_predictions as _load_predictions)
from paths import ANNOTATIONS_DIR, DATA_ROOT, REPO_ROOT  # noqa: E402


_PUB = REPO_ROOT
_BASE = DATA_ROOT
_HOME = _BASE / "home"
_OUT  = _PUB / "results" / "protein_class"

# Import calc_enrichment and plot rcParams from variant_db_charts
from analysis.variant_db_charts import calc_enrichment



ANNOTATION_CSV = _PUB / "results" / "protein_class" / "protein_class_annotations.csv"
# Must match the all-data model used for Fig 5 (weights/MutPred-PPI.pt). The old,
# variant_dbs_classified/ trees hold SF-model predictions; mixing the two
# across panels is what this path previously did.
def _prediction_tsv(db):
    """One database's prediction TSV, in either supported layout.

    Inference writes `{DATA_ROOT}/{db}/mutpred_ppi_predictions.tsv`; the
    reproduction notebook collects them as `{db}_mutpred_ppi_predictions.tsv`.
    Only the collected layout was spelled here, so this figure could not find its
    input at all after a plain inference run -- the same defect that left the
    master CSV loading zero rows. Same resolver as
    `extract_variant_db_stats.prediction_tsv`.
    """
    collected = _PUB / "results" / "variant_dbs_all_data" / f"{db}_mutpred_ppi_predictions.tsv"
    if collected.exists():
        return collected
    return Path(DATA_ROOT) / db / "mutpred_ppi_predictions.tsv"


CLINVAR_TSV    = _prediction_tsv("clinvar")
GNOMAD_TSV     = _prediction_tsv("gnomad")
PATHOGENIC_PKL = ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl"

# rcParams now come from analysis.plot_style.apply(), so every figure in the
# paper shares one font size, axis weight and output resolution. This block
# was duplicated verbatim in two scripts and absent from the other thirteen.

EDGOTYPE_CLASSES = list(edgotypes.EDGOTYPES)
SCORE_THRESHOLD  = 0.5
N_BOOTSTRAP      = 100_000
MIN_VARIANTS     = 30
RANDOM_SEED      = 42

# Classes to plot (omit "other")
PROTEIN_CLASSES = [
    "kinase",
    "transcription_factor",
    "receptor_signaling",
    "ubiquitin_proteasome",
    "dna_rna_binding",
    "structural_cytoskeletal",
    "metabolic_enzyme",
]

CLASS_X_LABELS = {
    "kinase":                 "Kinase",
    "transcription_factor":   "TF",
    "receptor_signaling":     "Receptor /\nSignaling",
    "ubiquitin_proteasome":   "Ubiquitin /\nProteasome",
    "dna_rna_binding":        "DNA / RNA\nBinding",
    "structural_cytoskeletal":"Structural /\nCytoskeletal",
    "metabolic_enzyme":       "Metabolic\nEnzyme",
}

CLASS_COLORS = {
    "kinase":                 "#1565C0",
    "transcription_factor":   "#2E7D32",
    "receptor_signaling":     "#FF7043",
    "ubiquitin_proteasome":   "#6A1B9A",
    "dna_rna_binding":        "#00838F",
    "structural_cytoskeletal":"#4E342E",
    "metabolic_enzyme":       "#F9A825",
}


def load_annotations() -> dict[str, str]:
    df = pd.read_csv(ANNOTATION_CSV)
    return dict(zip(df["uniprot_id"], df["protein_class"]))


def load_predictions(tsv: Path,
                     drop_training: bool = True) -> dict[tuple[str, str], dict[str, float]]:
    """(uniprot, variant) -> {partner: score}, via the shared reader.

    This figure reads the raw prediction TSVs rather than the classified stratum
    tables, so the training-overlap exclusion that `classify_variant_dbs.py`
    applies for Fig 5/S8/S9 has to be applied here too -- otherwise the same
    variants would be excluded from some panels of the paper and not others.
    """
    grouped = group_by_variant(_load_predictions(str(tsv)))
    if not drop_training:
        return grouped
    known = training_overlap.training_variants_or_none()
    if known is None:
        print(f"  [warn] {tsv.name}: training table unavailable, so variants seen "
              f"in training are NOT excluded -- these are not the published numbers",
              flush=True)
        return grouped
    kept = {k: v for k, v in grouped.items() if k not in known}
    dropped = len(grouped) - len(kept)
    if dropped:
        print(f"  training-overlap filter [{tsv.stem}]: dropped {dropped:,} of "
              f"{len(grouped):,} variants ({100 * dropped / len(grouped):.2f}%)",
              flush=True)
    return kept


def edgotypes_and_uniprots(
    grouped: dict[tuple[str, str], dict[str, float]],
    subset_by_vt: dict[tuple[str, str], set[str]] | None,
) -> tuple[list[str], list[str]]:
    ecs, unis = [], []
    for (u, v), pscores in grouped.items():
        if subset_by_vt is not None:
            if (u, v) not in subset_by_vt:
                continue
            scores = [pscores[p] for p in subset_by_vt[(u, v)] if p in pscores]
        else:
            scores = list(pscores.values())
        if not scores:
            continue
        ecs.append(edgotypes.classify(scores))
        unis.append(u)
    return ecs, unis


def multinomial_boot(counts: list[int], n: int) -> np.ndarray:
    """(n_bootstrap, 3) fraction array — same method as variant_db_charts.py."""
    total = sum(counts)
    probs = np.array(counts, dtype=float) / total
    return np.random.multinomial(total, probs, size=n) / total


def draw_enrichment_panel(ax, enrichment_boot_per_class, class_counts,
                           plotted, comp_idx, comp_name,
                           plot_idx, n_bonf, is_top):
    """Draw one enrichment panel — mirrors draw_component() in variant_db_charts.py."""
    alpha_bonf = 0.05 / n_bonf

    x_pos = 0
    x_ticks, x_labels = [], []

    for xi, pc in enumerate(plotted):
        vals   = enrichment_boot_per_class[pc][:, comp_idx]
        median = float(np.median(vals))
        p16    = float(np.percentile(vals, 16))
        p84    = float(np.percentile(vals, 84))

        if median >= 0:
            sig_bonf  = float(np.percentile(vals, 100 * alpha_bonf)) > 0
            sig_uncorr = float(np.percentile(vals, 5)) > 0
        else:
            sig_bonf  = float(np.percentile(vals, 100 * (1 - alpha_bonf))) < 0
            sig_uncorr = float(np.percentile(vals, 95)) < 0

        ax.bar(x_pos, median, width=0.8, color=CLASS_COLORS.get(pc, "#666"),
               edgecolor="black", linewidth=1.2, alpha=0.9)
        ax.errorbar(x_pos, median,
                    yerr=[[median - p16], [p84 - median]],
                    fmt="none", ecolor="black", capsize=4, linewidth=2, alpha=0.7)

        if sig_bonf:
            y_mark = (p84 + 0.05) if median > 0 else (p16 - 0.05)
            ax.text(x_pos, y_mark, "*", ha="center",
                    va="bottom" if median > 0 else "top",
                    fontsize=20, fontweight="bold", color="black")
        elif sig_uncorr:
            y_mark = (p84 + 0.03) if median > 0 else (p16 - 0.03)
            ax.text(x_pos, y_mark, "•", ha="center",
                    va="bottom" if median > 0 else "top",
                    fontsize=12, fontweight="bold", color="black")

        n = sum(class_counts[pc])
        x_labels.append(f"{CLASS_X_LABELS.get(pc, pc)}\n(n={n:,})")
        x_ticks.append(x_pos)
        x_pos += 1

    # Group boundary label (matching variant_db_charts.py style)
    mid = (len(plotted) - 1) / 2
    y_text = 0.90 if is_top else 1.02
    ax.text(mid, y_text, "ClinVar Pathogenic (GO class of interactor)",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                      edgecolor="gray", linewidth=1.5),
            transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_ylabel(f"{comp_name} Enrichment", fontsize=16, fontweight="bold")
    ax.set_xticks(x_ticks)

    if is_top:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=13)
        ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["top"].set_visible(False)

    ax.set_ylim(-1, 1.05)
    ax.set_xlim(-0.8, len(plotted) - 0.2)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)

    # Panel letter (A/B)
    ax.text(-0.12, 0.99, f"({chr(65 + plot_idx)})", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top", ha="right")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    print("Loading annotations...", flush=True)
    protein_class = load_annotations()

    print("Loading ClinVar pathogenic subset...", flush=True)
    with open(PATHOGENIC_PKL, "rb") as f:
        raw_subset = pickle.load(f)
    subset_by_vt: dict[tuple[str, str], set[str]] = defaultdict(set)
    for u, v, p in raw_subset:
        subset_by_vt[(u, v)].add(p)

    print("Loading ClinVar predictions...", flush=True)
    clinvar = load_predictions(CLINVAR_TSV)

    print("Loading gnomAD predictions...", flush=True)
    gnomad = load_predictions(GNOMAD_TSV)

    # gnomAD baseline — ALL variants
    print("Computing gnomAD baseline...", flush=True)
    gnomad_ecs, _ = edgotypes_and_uniprots(gnomad, subset_by_vt=None)
    gnomad_counts  = [sum(1 for e in gnomad_ecs if e == ec) for ec in EDGOTYPE_CLASSES]
    gnomad_boot    = multinomial_boot(gnomad_counts, N_BOOTSTRAP)
    print(f"  gnomAD n={len(gnomad_ecs):,}  {dict(zip(EDGOTYPE_CLASSES, gnomad_counts))}", flush=True)

    # ClinVar pathogenic — split by protein class
    print("Computing ClinVar pathogenic edgotypes...", flush=True)
    path_ecs, path_unis = edgotypes_and_uniprots(clinvar, subset_by_vt)
    print(f"  n={len(path_ecs):,}", flush=True)

    class_ecs: dict[str, list[str]] = defaultdict(list)
    for ec, uid in zip(path_ecs, path_unis):
        class_ecs[protein_class.get(uid, "other")].append(ec)

    # Bootstrap enrichment per class
    enrichment_boot: dict[str, np.ndarray] = {}
    class_counts: dict[str, list[int]] = {}

    for pc in PROTEIN_CLASSES:
        ecs = class_ecs.get(pc, [])
        if len(ecs) < MIN_VARIANTS:
            print(f"  [SKIP] {pc}: n={len(ecs)}", flush=True)
            continue
        counts = [sum(1 for e in ecs if e == ec) for ec in EDGOTYPE_CLASSES]
        class_counts[pc] = counts
        obs_boot = multinomial_boot(counts, N_BOOTSTRAP)
        enrichment_boot[pc] = np.array([
            calc_enrichment(obs_boot[i], gnomad_boot[i])
            for i in range(N_BOOTSTRAP)
        ])
        print(f"  {pc}: n={len(ecs):,}  {dict(zip(EDGOTYPE_CLASSES, counts))}", flush=True)

    plotted   = [pc for pc in PROTEIN_CLASSES if pc in enrichment_boot]
    n_bonf    = len(plotted) * 2   # 2 components tested per class
    scaled_w  = max(8, 1.6 * len(plotted))

    component_names   = ["Quasi-Null", "Edgetic"]
    component_indices = [1, 2]   # into EDGOTYPE_CLASSES

    fig, axes = plt.subplots(
        2, 1, figsize=(scaled_w, 10),
        gridspec_kw={"hspace": 0, "height_ratios": [1, 1]},
    )
    for plot_idx, (ci, comp_name) in enumerate(zip(component_indices, component_names)):
        draw_enrichment_panel(
            axes[plot_idx], enrichment_boot, class_counts, plotted,
            ci, comp_name, plot_idx, n_bonf, is_top=(plot_idx == 0),
        )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    out = _OUT / "pathogenic_by_class.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")

    # TSV
    rows = []
    for pc in plotted:
        n = sum(class_counts[pc])
        for ec_idx, ec in enumerate(EDGOTYPE_CLASSES):
            e = enrichment_boot[pc][:, ec_idx]
            rows.append({"protein_class": pc, "edgotype": ec, "n": n,
                         "median": float(np.median(e)),
                         "p16": float(np.percentile(e, 16)),
                         "p84": float(np.percentile(e, 84))})
    pd.DataFrame(rows).to_csv(_OUT / "pathogenic_by_class.tsv", sep="\t", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
