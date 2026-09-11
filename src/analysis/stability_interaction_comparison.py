#!/usr/bin/env python3
"""Compare MutPred-PPI interaction disruption scores vs stability ΔΔG.

For each variant database, plots interaction disruption (MutPred-PPI score)
against stability change (ΔΔG from pretrained MegaScale stability model).

Hypothesis: COSMIC cancer mutations may prefer to disrupt interactions while
preserving protein stability (interaction-specific mechanism), whereas
ClinVar pathogenic mutations may both disrupt interactions and destabilize.

Usage:
    conda run -n ppi python src/analysis/stability_interaction_comparison.py

Requirements:
  - results/variant_dbs_all_data/clinvar_mutpred_ppi_predictions.tsv
  - results/variant_dbs_stability/clinvar_stability_predictions.tsv
  (same for cosmic, gnomad)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --- repo-relative path resolution (see src/paths.py) ---
from paths import REPO_ROOT  # noqa: E402
from analysis.roc_plots import StaleCacheError  # noqa: E402


_PUB = REPO_ROOT
_DB_DIR  = _PUB / "results" / "variant_dbs_all_data"
_STAB    = _PUB / "results" / "variant_dbs_stability"
_OUT_DIR = _PUB / "results" / "stability_interaction"

DATASETS = {
    "clinvar": {
        "predictions": "clinvar_mutpred_ppi_predictions.tsv",
        "stability":   "clinvar_stability_predictions.tsv",
        "label":       "ClinVar",
        "color":       "#d62728",
        "filter":      lambda df: df,
    },
    "cosmic": {
        "predictions": "cosmic_mutpred_ppi_predictions.tsv",
        "stability":   "cosmic_stability_predictions.tsv",
        "label":       "COSMIC",
        "color":       "#ff7f0e",
        "filter":      lambda df: df,
    },
    "gnomad": {
        "predictions": "gnomad_mutpred_ppi_predictions.tsv",
        "stability":   "gnomad_stability_predictions.tsv",
        "label":       "gnomAD",
        "color":       "#2ca02c",
        "filter":      lambda df: df,
    },
    "hgmd": {
        "predictions": "hgmd_mutpred_ppi_predictions.tsv",
        "stability":   "hgmd_stability_predictions.tsv",
        "label":       "HGMD",
        "color":       "#9467bd",
        "filter":      lambda df: df,
    },
    "neurodev": {
        "predictions": "neurodev_mutpred_ppi_predictions.tsv",
        "stability":   "neurodev_stability_predictions.tsv",
        "label":       "Autism",
        "color":       "#8c564b",
        "filter":      lambda df: df,
    },
}


_NEW_PRED_COLS  = {"interactor", "partner", "mutation", "score"}
_NEW_STAB_COLS  = {"interactor", "partner", "mutation", "ddg_kcalmol"}


def _check_schema(path: Path, expected_new_cols: set[str]) -> None:
    """Raise StaleCacheError if the TSV header matches the old composite-id schema."""
    with open(path) as f:
        header = f.readline().rstrip("\n")
    cols = set(header.split("\t"))
    if "complex_id" in cols:
        raise StaleCacheError(
            f"{path.name} still uses the old `complex_id` schema. "
            f"Regenerate it with the migrated inference scripts "
            f"(`run_variant_db_inference.py` / `run_stability_inference.py`) "
            f"which emit `interactor / partner / mutation / score|ddg_kcalmol`.")
    if not expected_new_cols.issubset(cols):
        raise StaleCacheError(
            f"{path.name} header {header!r} lacks expected columns "
            f"{expected_new_cols - cols}.")


def load_joined(pred_tsv: Path, stab_tsv: Path) -> pd.DataFrame:
    """Join MutPred-PPI predictions with stability predictions on (interactor, partner, mutation).

    Both files must use the new explicit-column schema
    (`interactor / partner / mutation / score|ddg_kcalmol`). If either still carries
    the old `complex_id` schema, raises StaleCacheError naming which file must be
    regenerated — all existing on-disk TSVs are pre-migration and must be regenerated.
    """
    if not pred_tsv.exists():
        print(f"[SKIP] Missing: {pred_tsv}")
        return pd.DataFrame()
    if not stab_tsv.exists():
        print(f"[SKIP] Missing stability: {stab_tsv}")
        return pd.DataFrame()

    _check_schema(pred_tsv, _NEW_PRED_COLS)
    _check_schema(stab_tsv, _NEW_STAB_COLS)

    pred = pd.read_csv(pred_tsv, sep="\t")
    stab = pd.read_csv(stab_tsv, sep="\t")

    # Join on the canonical (interactor, partner, mutation) triple.
    merged = pred.merge(stab, on=["interactor", "partner", "mutation"], how="inner")
    print(f"  Joined: {len(merged):,} variants (pred={len(pred):,}, stab={len(stab):,})")
    return merged


def plot_scatter_grid(datasets_data: dict[str, pd.DataFrame]) -> None:
    """Scatter plot grid: interaction score vs stability ΔΔG per database."""
    available = {k: v for k, v in datasets_data.items() if not v.empty}
    n = len(available)
    if n == 0:
        print("No data to plot")
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax_idx, (ds_key, df) in enumerate(available.items()):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        cfg = DATASETS[ds_key]

        score = df["score"].values
        ddg   = df["ddg_kcalmol"].values

        # Hexbin for density
        hb = ax.hexbin(score, ddg, gridsize=50, cmap="YlOrRd", mincnt=1,
                       bins="log", alpha=0.9)
        fig.colorbar(hb, ax=ax, label="log(count)")

        # Spearman correlation
        rho, pval = stats.spearmanr(score, ddg)
        ax.set_title(f"{cfg['label']}\nSpearman ρ={rho:.3f}, n={len(df):,}", fontsize=10)
        ax.set_xlabel("MutPred-PPI score (interaction disruption)")
        ax.set_ylabel("ΔΔG stability (kcal/mol)")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")

    # Hide unused axes
    for ax_idx in range(n, nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    plt.suptitle("Interaction Disruption vs Stability: MutPred-PPI vs MegaScale Model",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / "interaction_vs_stability_scatter.png"
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot → {out}")


def plot_score_distributions(datasets_data: dict[str, pd.DataFrame]) -> None:
    """2D histogram / quadrant analysis: fraction in each quadrant per dataset."""
    available = {k: v for k, v in datasets_data.items() if not v.empty}
    if not available:
        return

    # Quadrant thresholds
    int_thresh = 0.5   # MutPred-PPI score threshold for "disrupted"
    stab_thresh = 0.5  # ΔΔG threshold for "destabilizing" (kcal/mol)

    rows = []
    for ds_key, df in available.items():
        cfg = DATASETS[ds_key]
        disrupted = df["score"] >= int_thresh
        destab    = df["ddg_kcalmol"] >= stab_thresh

        q_both  = int((disrupted & destab).sum())
        q_int   = int((disrupted & ~destab).sum())
        q_stab  = int((~disrupted & destab).sum())
        q_none  = int((~disrupted & ~destab).sum())
        total   = len(df)

        rows.append({
            "dataset":          cfg["label"],
            "n_total":          total,
            "n_both":           q_both,
            "pct_both":         100 * q_both / total,
            "n_int_only":       q_int,
            "pct_int_only":     100 * q_int / total,
            "n_stab_only":      q_stab,
            "pct_stab_only":    100 * q_stab / total,
            "n_neither":        q_none,
            "pct_neither":      100 * q_none / total,
            "median_score":     float(np.median(df["score"])),
            "median_ddg":       float(np.median(df["ddg_kcalmol"])),
        })
        print(f"  {cfg['label']}: disrupted+destab={q_both/total:.1%}, "
              f"int-only={q_int/total:.1%}, stab-only={q_stab/total:.1%}")

    summary_df = pd.DataFrame(rows)
    out_tsv = _OUT_DIR / "interaction_vs_stability_summary.tsv"
    summary_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved summary → {out_tsv}")

    # Stacked bar chart of quadrant fractions
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    width = 0.6

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#aec7e8"]
    labels_q = ["Disrupted + Destabilized", "Int. disrupted only",
                 "Destabilized only", "Neither"]
    cols = ["pct_both", "pct_int_only", "pct_stab_only", "pct_neither"]

    bottom = np.zeros(len(rows))
    for col, color, lbl in zip(cols, colors, labels_q):
        vals = summary_df[col].values
        ax.bar(x, vals, width, bottom=bottom, label=lbl, color=color, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([r["dataset"] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Fraction of variants (%)")
    ax.set_title("Interaction disruption vs stability quadrants\n"
                 f"(Int. thresh={int_thresh}, Stab. thresh={stab_thresh} kcal/mol)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out_bar = _OUT_DIR / "interaction_vs_stability_quadrants.png"
    plt.savefig(out_bar, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved quadrant bar chart → {out_bar}")


def main() -> None:
    datasets_data: dict[str, pd.DataFrame] = {}

    for ds_key, cfg in DATASETS.items():
        print(f"\nLoading {ds_key}...")
        pred_tsv = _DB_DIR / cfg["predictions"]
        stab_tsv = _STAB  / cfg["stability"]
        df = load_joined(pred_tsv, stab_tsv)
        if not df.empty:
            df = cfg["filter"](df)
        datasets_data[ds_key] = df

    print("\nGenerating scatter plot grid...")
    plot_scatter_grid(datasets_data)

    print("\nGenerating quadrant analysis...")
    plot_score_distributions(datasets_data)


if __name__ == "__main__":
    main()
