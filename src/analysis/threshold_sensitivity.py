#!/usr/bin/env python
"""Threshold sensitivity analysis for Reviewer 1.

Shows enrichment trends are robust across classification thresholds.
Same layout as enrichment_bootstrap_analysis_sufficient_partners.png but
with one set of overlaid semi-transparent bars per threshold
(0.3, 0.4, 0.5, 0.6, 0.7) on the same axes.

Reads pre-computed posterior_ls.pkl files (list of per-variant partner score
lists) from the classified variant DB directory, reclassifies at each threshold,
then runs multinomial bootstrap enrichment vs gnomAD.

Output:
  results_revisions/robustness_analyses/threshold_sensitivity.png
  results_revisions/robustness_analyses/threshold_sensitivity.tsv

Usage:
  conda run -n ppi python src/analysis/threshold_sensitivity.py [--n-bootstrap 10000]
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import REPO_ROOT  # noqa: E402


# ── Paths ──────────────────────────────────────────────────────────────────────
_PUB = str(REPO_ROOT)
DATA_DIR = f"{_PUB}/results_revisions/variant_dbs_sfvfp"
OUT_DIR = f"{_PUB}/results_revisions/robustness_analyses"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Color scheme: higher t = more stringent disruption threshold (disease mechanism).
# Blue (permissive, low t) → grey (reference, t=0.5) → red (stringent, high t).
# Follows RdBu diverging palette with medium grey at center to avoid white.
THRESHOLD_COLORS = {
    0.1: "#053061",   # very dark blue  — extremely permissive
    0.2: "#2166ac",   # dark blue
    0.3: "#4393c3",   # medium blue
    0.4: "#92c5de",   # light blue
    0.5: "#636363",   # medium grey     — reference threshold
    0.6: "#f4a582",   # light red-orange
    0.7: "#d6604d",   # medium red
    0.8: "#b2182b",   # dark red
    0.9: "#67001f",   # very dark red   — extremely stringent
}

# ── Group definitions (posterior_ls files) ─────────────────────────────────────
# Each entry: (display_group_key, pkl_path, x_label, bar_color)
# Groups and order matching enrichment_bootstrap_sufficient_partners layout

GROUPS = [
    # ClinVar
    ("clinvar_rare_benign",  f"{DATA_DIR}/clinvar/rare_benign_posterior_ls.pkl",
     "Rare Benign\n(ClinVar)", "#1565C0"),
    ("clinvar_benign",       f"{DATA_DIR}/clinvar/benign_posterior_ls.pkl",
     "Benign\n(ClinVar)", "#1976D2"),
    ("clinvar_pathogenic",   f"{DATA_DIR}/clinvar/pathogenic_posterior_ls.pkl",
     "Pathogenic\n(ClinVar)", "#D32F2F"),
    ("clinvar_vus",          f"{DATA_DIR}/clinvar/vus_posterior_ls.pkl",
     "VUS\n(ClinVar)", "#9E9E9E"),
    ("clinvar_ar_pathogenic", f"{DATA_DIR}/clinvar/ar_pathogenic_posterior_ls.pkl",
     "Pathogenic AR\n(ClinVar)", "#00897B"),
    ("clinvar_ad_pathogenic", f"{DATA_DIR}/clinvar/ad_pathogenic_posterior_ls.pkl",
     "Pathogenic AD\n(ClinVar)", "#FFB300"),
    # COSMIC
    ("cosmic_single",        f"{DATA_DIR}/cosmic/cosmic_single_posterior_ls.pkl",
     "Single\n(COSMIC)", "#FDD835"),
    ("cosmic_2+",            f"{DATA_DIR}/cosmic/cosmic_2+_posterior_ls.pkl",
     "≥2\n(COSMIC)", "#FFCA28"),
    ("cosmic_4+",            f"{DATA_DIR}/cosmic/cosmic_4+_posterior_ls.pkl",
     "≥4\n(COSMIC)", "#FF8F00"),
    ("cosmic_8+",            f"{DATA_DIR}/cosmic/cosmic_8+_posterior_ls.pkl",
     "≥8\n(COSMIC)", "#FF5722"),
    ("cosmic_16+",           f"{DATA_DIR}/cosmic/cosmic_16+_posterior_ls.pkl",
     "≥16\n(COSMIC)", "#E64A19"),
    ("cosmic_32+",           f"{DATA_DIR}/cosmic/cosmic_32+_posterior_ls.pkl",
     "≥32\n(COSMIC)", "#B71C1C"),
    # HGMD
    ("hgmd",                 f"{DATA_DIR}/hgmd/hgmd_posterior_ls.pkl",
     "HGMD", "#E74C3C"),
    ("hgmd_ar",              f"{DATA_DIR}/hgmd/ar_hgmd_posterior_ls.pkl",
     "AR\n(HGMD)", "#00897B"),
    ("hgmd_ad",              f"{DATA_DIR}/hgmd/ad_hgmd_posterior_ls.pkl",
     "AD\n(HGMD)", "#FFB300"),
    # gnomAD AF bins
    ("gnomad_af1",           f"{DATA_DIR}/gnomad/gnomad_upper_af_1e-06_posterior_ls.pkl",
     "AF≤1e-6\n(gnomAD)", "#FF7043"),
    ("gnomad_af2",           f"{DATA_DIR}/gnomad/gnomad_upper_af_1e-05_posterior_ls.pkl",
     "1e-6<AF≤1e-5\n(gnomAD)", "#FFA726"),
    ("gnomad_af3",           f"{DATA_DIR}/gnomad/gnomad_upper_af_0.0001_posterior_ls.pkl",
     "1e-5<AF≤1e-4\n(gnomAD)", "#FFCA28"),
    ("gnomad_af4",           f"{DATA_DIR}/gnomad/gnomad_upper_af_0.001_posterior_ls.pkl",
     "1e-4<AF≤1e-3\n(gnomAD)", "#9CCC65"),
    ("gnomad_af5",           f"{DATA_DIR}/gnomad/gnomad_upper_af_0.01_posterior_ls.pkl",
     "1e-3<AF≤1e-2\n(gnomAD)", "#66BB6A"),
    ("gnomad_af6",           f"{DATA_DIR}/gnomad/gnomad_upper_af_0.1_posterior_ls.pkl",
     "1e-2<AF\n(gnomAD)", "#4CAF50"),
    # NDD
    ("ndd_case",             f"{DATA_DIR}/neurodev/ndd_case_posterior_ls.pkl",
     "NDD Case", "#9C27B0"),
    ("ndd_control",          f"{DATA_DIR}/neurodev/ndd_control_posterior_ls.pkl",
     "NDD Control", "#607D8B"),
    # ASD
    ("fu_autism",            f"{DATA_DIR}/fu_autism/fu_autism_posterior_ls.pkl",
     "ASD Case", "#FF9800"),
]

# gnomAD overall (reference)
GNOMAD_ALL_PKL = f"{DATA_DIR}/gnomad/gnomad_posterior_ls.pkl"

# Visual separators between dataset groups (after these group keys, add spacing)
SEPARATOR_AFTER = {
    "clinvar_ad_pathogenic", "cosmic_32+", "hgmd_ad", "gnomad_af6", "ndd_control",
}


# ── Core helpers ───────────────────────────────────────────────────────────────

def classify_posterior_ls(posterior_ls, threshold):
    """Reclassify each variant's score list at the given threshold.

    Returns array of counts [n_quasi_null, n_edgetic, n_quasi_wt].
    """
    qn = e = qwt = 0
    for scores in posterior_ls:
        n_dis = sum(s > threshold for s in scores)
        n = len(scores)
        if n_dis == n:
            qn += 1
        elif n_dis == 0:
            qwt += 1
        else:
            e += 1
    return np.array([qn, e, qwt], dtype=float)


def bootstrap_enrichment(obs_counts, ref_counts, n_bootstrap, rng):
    """Multinomial bootstrap; returns (n_bootstrap, 3) array of enrichments."""
    n = obs_counts.sum()
    n_ref = ref_counts.sum()
    if n == 0 or n_ref == 0:
        return np.full((n_bootstrap, 3), np.nan)
    p_obs = obs_counts / n
    p_ref = ref_counts / n_ref
    boot_obs = rng.multinomial(n, p_obs, size=n_bootstrap) / n
    boot_ref = rng.multinomial(n_ref, p_ref, size=n_bootstrap) / n_ref
    # Enrichment = (f_obs - f_ref) / (f_obs + f_ref)
    denom = boot_obs + boot_ref
    denom = np.where(denom == 0, 1e-9, denom)
    return (boot_obs - boot_ref) / denom


def calc_enrichment(f_obs, f_ref):
    denom = f_obs + f_ref
    denom = np.where(denom == 0, 1e-9, denom)
    return (f_obs - f_ref) / denom


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # Load all posterior_ls files once
    print("Loading posterior_ls data...")
    group_data = {}
    for gkey, pkl_path, label, color in GROUPS:
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                group_data[gkey] = pickle.load(f)
            print(f"  {gkey}: n={len(group_data[gkey])}")
        else:
            print(f"  WARNING: {pkl_path} not found")

    if not os.path.exists(GNOMAD_ALL_PKL):
        raise FileNotFoundError(f"gnomAD reference not found: {GNOMAD_ALL_PKL}")
    with open(GNOMAD_ALL_PKL, "rb") as f:
        gnomad_all = pickle.load(f)
    print(f"  gnomad_all (reference): n={len(gnomad_all)}")

    # Compute bootstrap enrichment for each threshold and group
    # results[threshold][gkey] = (n_bootstrap, 3) enrichment array (QN, E, QWT)
    results = {}
    tsv_rows = []

    for threshold in THRESHOLDS:
        print(f"\nThreshold {threshold}...")
        results[threshold] = {}
        ref_counts = classify_posterior_ls(gnomad_all, threshold)

        for gkey, pkl_path, label, color in GROUPS:
            if gkey not in group_data:
                continue
            obs_counts = classify_posterior_ls(group_data[gkey], threshold)
            boot = bootstrap_enrichment(obs_counts, ref_counts, args.n_bootstrap, rng)
            results[threshold][gkey] = boot
            n = int(obs_counts.sum())
            for ci, comp in enumerate(["Quasi-null", "Edgetic", "Quasi-wild-type"]):
                m = float(np.median(boot[:, ci]))
                lo = float(np.percentile(boot[:, ci], 16))
                hi = float(np.percentile(boot[:, ci], 84))
                tsv_rows.append(f"{threshold}\t{gkey}\t{comp}\t{n}\t{m:.4f}\t{lo:.4f}\t{hi:.4f}")
            print(f"  {gkey}: n={n} | QN={int(obs_counts[0])} E={int(obs_counts[1])} QWT={int(obs_counts[2])}")

    # Save TSV
    out_tsv = os.path.join(OUT_DIR, "threshold_sensitivity.tsv")
    with open(out_tsv, "w") as f:
        f.write("threshold\tgroup\tedgotype\tn_variants\tmedian_enrichment\tlo68\thi68\n")
        f.write("\n".join(tsv_rows) + "\n")
    print(f"\nSaved: {out_tsv}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    # Build ordered x positions (same as enrichment bootstrap figure)
    groups_in_order = [(gkey, label, color) for gkey, _, label, color in GROUPS
                       if gkey in group_data]

    x_ticks = []
    x_labels = []
    x_separator_positions = []
    x_pos = 0
    for i, (gkey, label, color) in enumerate(groups_in_order):
        n = len(group_data[gkey])
        x_ticks.append(x_pos)
        x_labels.append(f"{label}\n(n={n:,})")
        if gkey in SEPARATOR_AFTER and i < len(groups_in_order) - 1:
            x_separator_positions.append(x_pos + 0.5)
            x_pos += 1.5
        else:
            x_pos += 1

    n_groups = len(x_ticks)
    n_thresh = len(THRESHOLDS)
    bar_w = 0.8 / n_thresh  # total bar width = 0.8, split across thresholds
    offsets = np.linspace(-0.4 + bar_w / 2, 0.4 - bar_w / 2, n_thresh)

    fig, axes = plt.subplots(2, 1, figsize=(max(len(groups_in_order) * 1.1, 16), 10),
                              gridspec_kw={"hspace": 0, "height_ratios": [1, 1]})

    comp_names = ["Quasi-Null", "Edgetic"]
    comp_indices = [0, 1]  # indices into the (QN, E, QWT) axis

    for pi, (comp_idx, comp_name) in enumerate(zip(comp_indices, comp_names)):
        ax = axes[pi]

        for ti, threshold in enumerate(THRESHOLDS):
            t_color = THRESHOLD_COLORS[threshold]
            alpha = 0.88 if threshold == 0.5 else 0.60

            for xi, (gkey, label, color) in enumerate(groups_in_order):
                if gkey not in results[threshold]:
                    continue
                boot = results[threshold][gkey]
                vals = boot[:, comp_idx]
                median = float(np.median(vals))
                p16 = float(np.percentile(vals, 16))
                p84 = float(np.percentile(vals, 84))
                xp = x_ticks[xi] + offsets[ti]
                edgecolor = "black" if threshold == 0.5 else "none"
                lw = 1.2 if threshold == 0.5 else 0.0
                ax.bar(xp, median, width=bar_w * 0.92, color=t_color,
                       edgecolor=edgecolor, linewidth=lw, alpha=alpha)
                ax.errorbar(xp, median, yerr=[[median - p16], [p84 - median]],
                            fmt="none", ecolor="black", capsize=1.5, linewidth=0.8, alpha=0.35)

        # Reference line
        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

        # Separator lines
        for sep_x in x_separator_positions:
            ax.axvline(sep_x, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

        ax.set_ylabel(f"{comp_name} Enrichment", fontsize=14, fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.set_ylim([-1, 1.05])
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)

        if pi == 0:
            # Legend for thresholds: blue (permissive) → grey (reference) → red (stringent)
            handles = [plt.Rectangle((0, 0), 1, 1,
                                     facecolor=THRESHOLD_COLORS[t],
                                     alpha=0.85 if t == 0.5 else 0.55,
                                     edgecolor="black" if t == 0.5 else "none",
                                     linewidth=1.2 if t == 0.5 else 0.0)
                       for t in THRESHOLDS]
            labels = [f"t={t:.1f}" + (" (ref)" if t == 0.5 else "") for t in THRESHOLDS]
            ax.legend(handles, labels,
                      title="Threshold  (low t = permissive; high t = stringent)",
                      loc="upper right", fontsize=8,
                      title_fontsize=8, ncol=9, framealpha=0.9)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", bottom=False, top=False)
            ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            ax.spines["bottom"].set_visible(False)
            ax.text(-0.035, 0.99, "(A)", transform=ax.transAxes,
                    fontsize=16, fontweight="bold", va="top", ha="right")
        else:
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=11)
            ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            ax.spines["top"].set_visible(False)
            ax.text(-0.035, 0.99, "(B)", transform=ax.transAxes,
                    fontsize=16, fontweight="bold", va="top", ha="right")

        ax.set_xlim(x_ticks[0] - 0.8, x_ticks[-1] + 0.8)

    plt.subplots_adjust(hspace=0)

    out_png = os.path.join(OUT_DIR, "threshold_sensitivity.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
