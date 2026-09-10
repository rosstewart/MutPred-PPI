#!/usr/bin/env python3
"""Scatter plot: max interaction disruption vs mean ΔΔG, per unique variant.

For each variant, aggregates across all tested partners:
  - max_score : max MutPred-PPI score across partners (worst-case disruption)
  - mean_ddg  : mean ΔΔG across partners (partner context affects GAT output slightly)

Groups (12 panels):
  ClinVar Pathogenic, ClinVar Benign, ClinVar VUS, HGMD, gnomAD,
  COSMIC highly recurrent (≥32 tumor sites),
  Oncogene / TSG (COSMIC recurrence ≥8 — too few onco/tsg variants at ≥32),
  AR-only / AD-only disease genes for ClinVar Pathogenic and HGMD
  (ClinGen Gene-Disease Validity MOI curations; see build_ar_ad_gene_sets.py)

Per-variant aggregation filters partners to exactly those listed in each group's
subset PKL (mirrors classify_variant_dbs.py::build_arrays) so sample sizes match
Fig 5 (variant_db_charts.py) exactly.

Usage:
    conda run -n ppi python src/analysis/stability_interaction_scatter.py [--cosmic-min-recurrence 32]

Output:
    results/stability_interaction/scatter_per_variant.png
    results/stability_interaction/scatter_per_variant_kde.png
    results/stability_interaction/per_variant_summary.tsv
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT, REPO_ROOT  # noqa: E402
from roc_plots import StaleCacheError  # noqa: E402
from utils import mutations  # noqa: E402


_PUB = REPO_ROOT
_BASE = DATA_ROOT
_HOME  = _BASE / "home"
# Must match the all-data model used for Fig 5 (weights/MutPred-PPI.pt). The old,
# variant_dbs_classified/ trees hold SF-model predictions; mixing the two
# across panels is what this path previously did.
_DB    = _PUB / "results" / "variant_dbs_all_data"
_STAB  = _PUB / "results" / "variant_dbs_stability"
_OUT   = _PUB / "results" / "stability_interaction"

ONCO_TSG_FILE = ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl"
AR_AD_FILE    = ANNOTATIONS_DIR / "clingen_ar_ad_uniprot_sets.pkl"
ONCO_TSG_MIN_RECURRENCE = 8  # too few onco/tsg variants at ≥32 (139/114)

SUBSET_PKLS = {
    "ClinVar Pathogenic": (ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl", "clinvar"),
    "ClinVar Benign":     (ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl",     "clinvar"),
    "ClinVar VUS":        (ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl",        "clinvar"),
    "HGMD":               (ANNOTATIONS_LICENSED_DIR / "hgmd_variant_subset.pkl",        "hgmd"),
}

# Fig 5 reference n's (variant_db_charts.py, results/variant_dbs_all_data) — for sanity check
FIG5_REFERENCE_N = {
    "ClinVar Pathogenic": 27843,
    "ClinVar Benign":     14579,
    "ClinVar VUS":        245146,
    "HGMD":               3478,
}

# gnomAD: no subset PKL, use all variants. Colors match the canonical palette in
# variant_db_charts.py::_get_enrich_color() for cross-figure consistency.
GROUP_COLORS = {
    "ClinVar Pathogenic":    "#D32F2F",
    "ClinVar Benign":        "#1976D2",
    "ClinVar VUS":           "#9E9E9E",
    "HGMD":                  "#E74C3C",
    "gnomAD":                "#2ca02c",
    "COSMIC recurrent":      "#B71C1C",
    "Onco":                  "#FF3D00",
    "TSG":                   "#FF6E40",
    "AR ClinVar Pathogenic": "#00897B",
    "AD ClinVar Pathogenic": "#FFB300",
    "AR HGMD":               "#00897B",
    "AD HGMD":               "#FFB300",
}


def _check_tsv_schema(path: Path, expected_cols: set[str]) -> None:
    """Raise StaleCacheError if the TSV still uses the old composite-id schema."""
    with open(path) as f:
        header = f.readline().rstrip("\n")
    cols = set(header.split("\t"))
    if "complex_id" in cols:
        raise StaleCacheError(
            f"{path.name} still uses the old `complex_id` schema. "
            f"Regenerate it with the migrated inference scripts "
            f"(`run_variant_db_inference.py` / `run_stability_inference.py`) "
            f"which emit `interactor / partner / mutation / score|ddg_kcalmol`.")
    if not expected_cols.issubset(cols):
        raise StaleCacheError(
            f"{path.name} header lacks expected columns {expected_cols - cols}.")


def load_tsv_grouped(pred_tsv: Path, stab_tsv: Path) -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Return {(interactor, variant_0b): [(partner, score, ddg), ...]} joined on explicit columns.

    Both files must use the new explicit-column schema. Existing on-disk TSVs use
    the old `complex_id` schema and will raise StaleCacheError until regenerated.

    Internal representation uses 0-based variant strings for consistency with the
    subset PKLs (which store 1-based variants; the conversion happens at the join in
    aggregate_per_variant). The `uniprot` key is the interactor accession.
    """
    if not pred_tsv.exists() or not stab_tsv.exists():
        return {}

    _check_tsv_schema(pred_tsv, {"interactor", "partner", "mutation", "score"})
    _check_tsv_schema(stab_tsv, {"interactor", "partner", "mutation", "ddg_kcalmol"})

    # Load stability: (interactor, partner, mutation_1b) -> ddg
    stab: dict[tuple[str, str, str], float] = {}
    stab_df = pd.read_csv(stab_tsv, sep="\t")
    for _, row in stab_df.iterrows():
        stab[(str(row["interactor"]), str(row["partner"]), str(row["mutation"]))] = float(row["ddg_kcalmol"])

    # Group (partner, score, ddg) tuples by (interactor, variant_0b).
    # partner identity is retained so aggregate_per_variant() can filter per-partner.
    result: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
    pred_df = pd.read_csv(pred_tsv, sep="\t")
    n_no_stab = 0
    for _, row in pred_df.iterrows():
        interactor = str(row["interactor"])
        partner    = str(row["partner"])
        mut_1b     = str(row["mutation"])
        score      = float(row["score"])
        ddg = stab.get((interactor, partner, mut_1b))
        if ddg is not None:
            var0 = mutations.to_zero_based(mut_1b)
            result[(interactor, var0)].append((partner, score, ddg))
        else:
            n_no_stab += 1
    if n_no_stab:
        print(f"  {n_no_stab:,} pred rows had no matching stability entry (skipped)", flush=True)

    return result


def aggregate_per_variant(
    grouped: dict[tuple[str, str], list[tuple[str, float, float]]],
    subset: set[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Aggregate to per-variant: max score, mean ΔΔG, restricted to partners in `subset`.

    subset: set of (uniprot, variant_1b, partner) — converted to 0-based variant key
    internally. Filtering is per-(variant,partner), mirroring
    classify_variant_dbs.py::build_arrays() — a variant qualifying via one partner
    must not pull in scores from OTHER partners not listed in the subset.
    """
    if subset is not None:
        allowed: dict[tuple[str, str], set[str]] = defaultdict(set)
        for u, v1b, p in subset:
            try:
                var0 = mutations.to_zero_based(str(v1b))
            except ValueError:
                continue  # skip malformed entries (e.g. accession in variant field)
            allowed[(u, var0)].add(p)
    else:
        allowed = None

    rows = []
    for (uniprot, var0), entries in grouped.items():
        if allowed is not None:
            allowed_partners = allowed.get((uniprot, var0))
            if not allowed_partners:
                continue
            entries = [(p, s, d) for (p, s, d) in entries if p in allowed_partners]
        if not entries:
            continue
        scores = [s for (_, s, _) in entries]
        ddgs   = [d for (_, _, d) in entries]
        rows.append({
            "uniprot":    uniprot,
            "variant":    var0,
            "max_score":  max(scores),
            "mean_score": np.mean(scores),
            "n_partners": len(scores),
            "mean_ddg":   np.mean(ddgs),
            "max_ddg":    max(ddgs),
        })
    return pd.DataFrame(rows)


def plot_scatter(groups: dict[str, pd.DataFrame], out: Path, sample_n: int = 5000) -> None:
    """Scatter plot with downsampling for dense groups."""
    fig, ax = plt.subplots(figsize=(8, 6))

    rng = np.random.default_rng(42)
    for label, df in groups.items():
        if df.empty:
            continue
        color = GROUP_COLORS.get(label, "grey")
        idx = rng.choice(len(df), size=min(sample_n, len(df)), replace=False)
        sub = df.iloc[idx]
        ax.scatter(sub["mean_ddg"], sub["max_score"],
                   c=color, alpha=0.15, s=4, label=f"{label} (n={len(df):,})", rasterized=True)

    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean ΔΔG across partners (kcal/mol)\n← stabilizing | destabilizing →")
    ax.set_ylabel("Max interaction disruption score across partners\n(MutPred-PPI, 0–1)")
    ax.set_title("Stability vs Interaction Disruption per Variant")
    ax.legend(fontsize=7, markerscale=4, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter → {out}")


def plot_kde_contours(groups: dict[str, pd.DataFrame], out: Path, cosmic_min_recurrence: int,
                       max_n: int = 20000) -> None:
    """KDE contour plot per group — better for seeing cluster shapes."""
    n_groups = len(groups)
    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    items = list(groups.items())
    rng = np.random.default_rng(42)

    # Determine axis limits from all data
    all_ddg = np.concatenate([df["mean_ddg"].values for df in groups.values() if not df.empty])
    all_score = np.concatenate([df["max_score"].values for df in groups.values() if not df.empty])
    xlim = (np.percentile(all_ddg, 1), np.percentile(all_ddg, 99))
    ylim = (0, 1)

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    for idx, (label, df) in enumerate(items):
        ax = axes_flat[idx]
        color = GROUP_COLORS.get(label, "grey")

        panel_letter = f"({chr(65 + idx)})"
        ax.text(-0.08, 1.18, panel_letter, transform=ax.transAxes,
                 fontsize=12, fontweight="bold", va="top")

        if label == "COSMIC recurrent":
            title = f"COSMIC recurrent (≥{cosmic_min_recurrence} tumor sites)"
        elif label in ("Onco", "TSG"):
            title = f"{label} (COSMIC recurrence ≥{ONCO_TSG_MIN_RECURRENCE})"
        else:
            title = label

        if df.empty:
            ax.set_title(f"{title}\n(no data)", fontsize=9)
            continue

        # Downsample for KDE
        n = min(max_n, len(df))
        idx_s = rng.choice(len(df), size=n, replace=False)
        x = df.iloc[idx_s]["mean_ddg"].values
        y = df.iloc[idx_s]["max_score"].values

        # KDE
        try:
            kernel = gaussian_kde(np.vstack([x, y]), bw_method=0.15)
            Z = kernel(positions).reshape(XX.shape)
            ax.contourf(XX, YY, Z, levels=12, cmap="Blues" if "gnomAD" in label or "Benign" in label else "Reds",
                        alpha=0.7)
            ax.contour(XX, YY, Z, levels=6, colors=color, linewidths=0.5, alpha=0.6)
        except Exception:
            ax.scatter(x, y, c=color, alpha=0.05, s=2, rasterized=True)

        ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.axvline(0.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.set_title(f"{title}\n(n={len(df):,} variants)", fontsize=9)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Median crosshair
        ax.axvline(df["mean_ddg"].median(), color=color, linewidth=1.5, linestyle=":",
                   alpha=0.8, label=f"median ΔΔG={df['mean_ddg'].median():.2f}")
        ax.axhline(df["max_score"].median(), color=color, linewidth=1.5, linestyle="-.",
                   alpha=0.8, label=f"median score={df['max_score'].median():.2f}")
        ax.legend(fontsize=6, loc="upper right")

    for idx in range(len(items), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Mean ΔΔG (kcal/mol)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Max disruption score", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved KDE → {out}")


def _cosmic_recurrence_set(vt_to_sites: dict[str, list[str]], min_recurrence: int) -> set[tuple[str, str]]:
    """{(uniprot, variant_0b)} for COSMIC entries with tumor-site recurrence >= min_recurrence.

    The COSMIC recurrence dict keys are `"ACC MUT_1b"` (space-delimited, 1-based mutation).
    Internal representation is 0-based for consistency with `load_tsv_grouped`.
    """
    out = set()
    for key, sites in vt_to_sites.items():
        if len(sites) >= min_recurrence:
            parts = key.split(" ", 1)
            if len(parts) == 2:
                u, v1b = parts
                try:
                    var0 = mutations.to_zero_based(v1b)
                except ValueError:
                    continue  # skip malformed entries (e.g. delins notation)
                out.add((u, var0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmic-min-recurrence", type=int, default=32,
                    help="Min tumor-site recurrence for 'COSMIC recurrent' group (default: 32, "
                         "matching Fig 5's highest recurrence bin)")
    args = ap.parse_args()

    _OUT.mkdir(parents=True, exist_ok=True)

    # --- Load COSMIC recurrence ---
    cosmic_rec_file = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"
    cosmic_high_rec: set[tuple[str, str]] | None = None
    cosmic_rec8: set[tuple[str, str]] | None = None
    if cosmic_rec_file.exists():
        print(f"Loading COSMIC recurrence (min={args.cosmic_min_recurrence})...", flush=True)
        with open(cosmic_rec_file, "rb") as f:
            vt_to_sites = pickle.load(f)
        cosmic_high_rec = _cosmic_recurrence_set(vt_to_sites, args.cosmic_min_recurrence)
        print(f"  {len(cosmic_high_rec):,} high-recurrence COSMIC variants", flush=True)
        cosmic_rec8 = _cosmic_recurrence_set(vt_to_sites, ONCO_TSG_MIN_RECURRENCE)
        print(f"  {len(cosmic_rec8):,} COSMIC variants at recurrence ≥{ONCO_TSG_MIN_RECURRENCE} "
              f"(for Onco/TSG)", flush=True)

    # --- Load Onco/TSG protein-level sets ---
    onco_uniprots: set[str] = set()
    tsg_uniprots: set[str] = set()
    if ONCO_TSG_FILE.exists():
        with open(ONCO_TSG_FILE, "rb") as f:
            onco_tsg = pickle.load(f)
        onco_uniprots = {v.split()[0] for v in onco_tsg.get("oncogene", set())}
        tsg_uniprots  = {v.split()[0] for v in onco_tsg.get("TSG", set())}
        print(f"Loaded onco_tsg_dict: {len(onco_uniprots)} oncogene, "
              f"{len(tsg_uniprots)} TSG UniProt IDs", flush=True)

    # --- Load AR/AD UniProt sets ---
    ar_uniprots: set[str] = set()
    ad_uniprots: set[str] = set()
    if AR_AD_FILE.exists():
        with open(AR_AD_FILE, "rb") as f:
            ar_ad = pickle.load(f)
        ar_uniprots = ar_ad.get("AR", set())
        ad_uniprots = ar_ad.get("AD", set())
        print(f"Loaded clingen_ar_ad_uniprot_sets: {len(ar_uniprots)} AR-only, "
              f"{len(ad_uniprots)} AD-only UniProt IDs", flush=True)
    else:
        print(f"[WARN] {AR_AD_FILE} not found — run build_ar_ad_gene_sets.py first; "
              "AR/AD groups will be empty", flush=True)

    # --- Load and group TSV data ---
    db_grouped: dict[str, dict] = {}
    for db in ["clinvar", "hgmd", "gnomad", "cosmic"]:
        pred_tsv = _DB  / f"{db}_mutpred_ppi_predictions.tsv"
        stab_tsv = _STAB / f"{db}_stability_predictions.tsv"
        if not pred_tsv.exists() or not stab_tsv.exists():
            print(f"[SKIP] {db}: missing TSV", flush=True)
            continue
        print(f"Loading {db}...", flush=True)
        db_grouped[db] = load_tsv_grouped(pred_tsv, stab_tsv)
        print(f"  {len(db_grouped[db]):,} unique (uniprot, variant) pairs", flush=True)

    # --- Build per-group DataFrames ---
    groups: dict[str, pd.DataFrame] = {}
    raw_subsets: dict[str, set[tuple[str, str, str]]] = {}

    for label, (pkl_path, db) in SUBSET_PKLS.items():
        if db not in db_grouped:
            continue
        if not pkl_path.exists():
            print(f"[SKIP] {label}: subset PKL not found", flush=True)
            continue
        print(f"Building {label}...", flush=True)
        with open(pkl_path, "rb") as f:
            subset = pickle.load(f)  # set of (uniprot, variant_1b, partner)
        raw_subsets[label] = subset
        groups[label] = aggregate_per_variant(db_grouped[db], subset)
        print(f"  {len(groups[label]):,} variants", flush=True)

    if "gnomad" in db_grouped:
        print("Building gnomAD (all)...", flush=True)
        groups["gnomAD"] = aggregate_per_variant(db_grouped["gnomad"], subset=None)
        print(f"  {len(groups['gnomAD']):,} variants", flush=True)

    if "cosmic" in db_grouped and cosmic_high_rec is not None:
        print(f"Building COSMIC recurrent (≥{args.cosmic_min_recurrence})...", flush=True)
        cosmic_all = aggregate_per_variant(db_grouped["cosmic"], subset=None)
        mask = cosmic_all.apply(lambda r: (r["uniprot"], r["variant"]) in cosmic_high_rec, axis=1)
        groups["COSMIC recurrent"] = cosmic_all[mask].reset_index(drop=True)
        print(f"  {len(groups['COSMIC recurrent']):,} variants", flush=True)

        if False and cosmic_rec8 is not None and (onco_uniprots or tsg_uniprots):
            # Excluded from the final 6-panel figure (Pathogenic/Benign/VUS/gnomAD/HGMD/COSMIC≥32 only).
            # Data-loading logic kept intact in case Onco/TSG panels are wanted again later.
            print(f"Building Onco (COSMIC recurrence ≥{ONCO_TSG_MIN_RECURRENCE})...", flush=True)
            onco_mask = cosmic_all.apply(
                lambda r: (r["uniprot"], r["variant"]) in cosmic_rec8 and r["uniprot"] in onco_uniprots,
                axis=1)
            groups["Onco"] = cosmic_all[onco_mask].reset_index(drop=True)
            print(f"  {len(groups['Onco']):,} variants", flush=True)

            print(f"Building TSG (COSMIC recurrence ≥{ONCO_TSG_MIN_RECURRENCE})...", flush=True)
            tsg_mask = cosmic_all.apply(
                lambda r: (r["uniprot"], r["variant"]) in cosmic_rec8 and r["uniprot"] in tsg_uniprots,
                axis=1)
            groups["TSG"] = cosmic_all[tsg_mask].reset_index(drop=True)
            print(f"  {len(groups['TSG']):,} variants", flush=True)

    if False and (ar_uniprots or ad_uniprots):
        # Excluded from the final 6-panel figure (Pathogenic/Benign/VUS/gnomAD/HGMD/COSMIC≥32 only).
        # Data-loading logic kept intact in case AR/AD panels are wanted again later.
        if "ClinVar Pathogenic" in raw_subsets and "clinvar" in db_grouped:
            pathogenic_subset = raw_subsets["ClinVar Pathogenic"]
            ar_path_subset = {(u, v, p) for (u, v, p) in pathogenic_subset if u in ar_uniprots}
            ad_path_subset = {(u, v, p) for (u, v, p) in pathogenic_subset if u in ad_uniprots}
            print("Building AR ClinVar Pathogenic...", flush=True)
            groups["AR ClinVar Pathogenic"] = aggregate_per_variant(db_grouped["clinvar"], ar_path_subset)
            print(f"  {len(groups['AR ClinVar Pathogenic']):,} variants", flush=True)
            print("Building AD ClinVar Pathogenic...", flush=True)
            groups["AD ClinVar Pathogenic"] = aggregate_per_variant(db_grouped["clinvar"], ad_path_subset)
            print(f"  {len(groups['AD ClinVar Pathogenic']):,} variants", flush=True)

        if "HGMD" in raw_subsets and "hgmd" in db_grouped:
            hgmd_subset = raw_subsets["HGMD"]
            ar_hgmd_subset = {(u, v, p) for (u, v, p) in hgmd_subset if u in ar_uniprots}
            ad_hgmd_subset = {(u, v, p) for (u, v, p) in hgmd_subset if u in ad_uniprots}
            print("Building AR HGMD...", flush=True)
            groups["AR HGMD"] = aggregate_per_variant(db_grouped["hgmd"], ar_hgmd_subset)
            print(f"  {len(groups['AR HGMD']):,} variants", flush=True)
            print("Building AD HGMD...", flush=True)
            groups["AD HGMD"] = aggregate_per_variant(db_grouped["hgmd"], ad_hgmd_subset)
            print(f"  {len(groups['AD HGMD']):,} variants", flush=True)

    # --- Sanity check against Fig 5 reference counts ---
    print("\nSanity check vs Fig 5 (variant_db_charts.py) reference counts:")
    for label, ref_n in FIG5_REFERENCE_N.items():
        actual_n = len(groups.get(label, pd.DataFrame()))
        mark = "OK" if actual_n == ref_n else "MISMATCH"
        print(f"  [{mark}] {label}: n={actual_n:,} (Fig 5 n={ref_n:,})")
    if args.cosmic_min_recurrence == 32 and "COSMIC recurrent" in groups:
        ref_n = 1673
        actual_n = len(groups["COSMIC recurrent"])
        mark = "OK" if actual_n == ref_n else "MISMATCH"
        print(f"  [{mark}] COSMIC recurrent (≥32): n={actual_n:,} (Fig 5 n={ref_n:,})")

    # --- Summary statistics ---
    print("\nPer-variant summary:")
    rows = []
    for label, df in groups.items():
        if df.empty:
            continue
        rows.append({
            "group": label,
            "n_variants": len(df),
            "median_max_score": df["max_score"].median(),
            "median_mean_ddg":  df["mean_ddg"].median(),
            "pct_score_gt05":   (df["max_score"] > 0.5).mean() * 100,
            "pct_ddg_gt05":     (df["mean_ddg"] > 0.5).mean() * 100,
            "pct_both":         ((df["max_score"] > 0.5) & (df["mean_ddg"] > 0.5)).mean() * 100,
        })
        print(f"  {label}: n={len(df):,}  "
              f"med_score={df['max_score'].median():.3f}  "
              f"med_ddg={df['mean_ddg'].median():.3f}")

    pd.DataFrame(rows).to_csv(_OUT / "per_variant_summary.tsv", sep="\t", index=False, float_format="%.4f")

    # --- Plots ---
    print("\nGenerating scatter...", flush=True)
    plot_scatter(groups, _OUT / "scatter_per_variant.png")
    print("Generating KDE contours...", flush=True)
    plot_kde_contours(groups, _OUT / "scatter_per_variant_kde.png", args.cosmic_min_recurrence)


if __name__ == "__main__":
    main()
