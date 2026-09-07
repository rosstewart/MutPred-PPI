#!/usr/bin/env python3
"""Data quality and properties comparison across VarChAMP dataset variants.

Compares: Sahni/Fragoza (baseline), VarChAMP2026, VarChAMP1p, CAVA, VarChAMP_pooled.

Outputs:
  - TSV summary table
  - Bar charts: disruption rate, unique proteins/complexes, multi-partner coverage

Usage:
    conda run -n ppi env OPENBLAS_NUM_THREADS=1 python \
        publication/src/analysis/varchamp_dataset_comparison.py \
        --out-dir publication/results_revisions/dataset_comparison/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import DATA_CACHES_DIR, REVISIONS_DIR  # noqa: E402


# ── constants ──────────────────────────────────────────────────────────────────

TRAINING_CSV = str(DATA_CACHES_DIR / "training_data_internal.csv")
SFVC2026_CSV = str(REVISIONS_DIR / "sfvc2026_labeled_data.csv")
DATASET_MAP = {
    "Sahni":          "Sahni",
    "Fragoza":        "Fragoza",
    "VarChAMP":       "VarChAMP1p",
    "VarChAMP_pooled": "VarChAMP_pooled",
}

COLORS = {
    "Sahni":           "#4C72B0",
    "Fragoza":         "#DD8452",
    "VarChAMP2026":    "#55A868",
    "VarChAMP1p":      "#C44E52",
    "CAVA":            "#8172B2",
    "VarChAMP_pooled": "#937860",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(TRAINING_CSV)
    return df


def filter_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return rows where dataset equals exactly source (not compound labels)."""
    return df[df["dataset"] == source].copy()


def filter_contains(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return rows where dataset contains source."""
    return df[df["dataset"].str.contains(source, na=False)].copy()


def compute_stats(sub: pd.DataFrame, name: str) -> dict:
    """Compute summary statistics for a subset of training_data.csv."""
    n_variants = len(sub)
    n_complexes = sub.groupby(["interactor", "partner"]).ngroups
    n_unique_interactors = sub["interactor"].nunique()
    n_unique_partners = sub["partner"].nunique()
    n_unique_proteins = len(set(sub["interactor"].tolist() + sub["partner"].tolist()))
    disruption_rate = sub["perturbed"].mean() if "perturbed" in sub.columns else float("nan")

    # Multi-partner: interactors tested against >1 partner
    partners_per_interactor = sub.groupby("interactor")["partner"].nunique()
    n_multi_partner_interactors = (partners_per_interactor > 1).sum()
    frac_multi_partner = n_multi_partner_interactors / max(len(partners_per_interactor), 1)

    # Variants per complex
    variants_per_complex = sub.groupby(["interactor", "partner"]).size()
    median_variants_per_complex = variants_per_complex.median()

    return {
        "dataset": name,
        "n_variants": n_variants,
        "n_complexes": n_complexes,
        "n_unique_proteins": n_unique_proteins,
        "n_unique_interactors": n_unique_interactors,
        "disruption_rate": round(disruption_rate * 100, 1),
        "n_multi_partner_interactors": n_multi_partner_interactors,
        "frac_multi_partner_pct": round(frac_multi_partner * 100, 1),
        "median_variants_per_complex": round(median_variants_per_complex, 1),
    }


def load_vc2026_stats(sfvc2026_csv: str) -> dict:
    """Load VarChAMP2026 stats from the separate labeled CSV."""
    try:
        df = pd.read_csv(sfvc2026_csv)
        vc_df = df[df["dataset"].str.contains("VarChAMP", na=False)].copy()
        if len(vc_df) == 0:
            return {}
        return compute_stats(vc_df, "VarChAMP2026")
    except FileNotFoundError:
        return {}


def conflict_analysis(df: pd.DataFrame) -> dict:
    """Count variants with label conflicts when merging sources."""
    label_per_vt: dict = {}
    conflict_count = 0

    for _, row in df.iterrows():
        vt_key = (row["interactor"], row["mutation"], row["partner"])
        label = bool(row["perturbed"])
        if vt_key in label_per_vt:
            if label_per_vt[vt_key] != label:
                conflict_count += 1
        else:
            label_per_vt[vt_key] = label

    return {"n_conflicting_variants": conflict_count, "n_unique_variants": len(label_per_vt)}


# ── plotting ──────────────────────────────────────────────────────────────────

def bar_chart(stats_list: list, metric: str, ylabel: str, title: str, out_path: str) -> None:
    names = [s["dataset"] for s in stats_list]
    values = [s.get(metric, 0) for s in stats_list]
    colors = [COLORS.get(n, "#888888") for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1f" if any(isinstance(v, float) for v in values) else "%.0f",
                 padding=3, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results_revisions/dataset_comparison")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading training data...", flush=True)
    df = load_training_data()
    print(f"  Total rows: {len(df)}", flush=True)

    # ── Per-source stats ──────────────────────────────────────────────────────
    stats_list = []

    for source_key, display_name in DATASET_MAP.items():
        sub = filter_source(df, source_key)
        if len(sub) > 0:
            s = compute_stats(sub, display_name)
            stats_list.append(s)
            print(f"  {display_name}: {len(sub)} rows", flush=True)

    # VarChAMP2026 from separate CSV
    vc2026_stats = load_vc2026_stats(SFVC2026_CSV)
    if vc2026_stats:
        stats_list.append(vc2026_stats)
        print(f"  VarChAMP2026: {vc2026_stats['n_variants']} rows", flush=True)

    # CAVA: check if present in training_data.csv
    cava_sub = filter_contains(df, "CAVA") if "CAVA" in df.get("dataset", pd.Series()).values else pd.DataFrame()
    if len(cava_sub) == 0:
        # Try from sfvc2026_labeled_data.csv
        try:
            sfvc_df = pd.read_csv(SFVC2026_CSV)
            cava_sub = sfvc_df[sfvc_df["dataset"].str.contains("CAVA", na=False)]
            if len(cava_sub) > 0:
                s = compute_stats(cava_sub, "CAVA")
                stats_list.append(s)
                print(f"  CAVA: {len(cava_sub)} rows (from sfvc2026_labeled_data.csv)", flush=True)
        except FileNotFoundError:
            pass

    # Order: Sahni, Fragoza, VarChAMP2026, VarChAMP1p, CAVA, VarChAMP_pooled
    order = ["Sahni", "Fragoza", "VarChAMP2026", "VarChAMP1p", "CAVA", "VarChAMP_pooled"]
    stats_map = {s["dataset"]: s for s in stats_list}
    stats_ordered = [stats_map[k] for k in order if k in stats_map]

    # ── Summary TSV ───────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(stats_ordered)
    tsv_path = out_dir / "dataset_comparison_summary.tsv"
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nSummary table saved: {tsv_path}", flush=True)
    print(summary_df.to_string(index=False))

    # ── Conflict analysis for combined datasets ────────────────────────────────
    print("\nConflict analysis (combined training_data.csv)...", flush=True)
    conflict_info = conflict_analysis(df)
    print(f"  Unique variants (interactor, mutation, partner): {conflict_info['n_unique_variants']}")
    print(f"  Conflicting labels: {conflict_info['n_conflicting_variants']}")

    # ── Bar charts ────────────────────────────────────────────────────────────
    print("\nGenerating charts...", flush=True)

    bar_chart(
        stats_ordered, "n_variants", "Number of variants", "Dataset size",
        str(out_dir / "chart_n_variants.png"),
    )
    bar_chart(
        stats_ordered, "disruption_rate", "Disruption rate (%)", "Disruption rate (% perturbed)",
        str(out_dir / "chart_disruption_rate.png"),
    )
    bar_chart(
        stats_ordered, "n_complexes", "Unique complexes", "Number of unique complexes",
        str(out_dir / "chart_n_complexes.png"),
    )
    bar_chart(
        stats_ordered, "frac_multi_partner_pct",
        "Interactors with >1 partner (%)", "Multi-partner coverage",
        str(out_dir / "chart_multi_partner.png"),
    )
    bar_chart(
        stats_ordered, "n_unique_proteins", "Unique proteins", "Unique proteins (interactors + partners)",
        str(out_dir / "chart_unique_proteins.png"),
    )

    print(f"\nAll outputs in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
