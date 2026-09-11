#!/usr/bin/env python3
"""Experimental clustering analysis: pool ClinVar Pathogenic, ClinVar Benign, HGMD,
and COSMIC recurrent (≥32 tumor sites) per-variant (mean_ddg, max_score) points and
cluster in 2D to test whether stability disruption implies interaction disruption
but not vice versa (i.e. expect a cluster with high interaction-disruption score but
low ΔΔG — "interaction-only" — but no cluster with high ΔΔG and low score).

NOT part of the manuscript — internal/exploratory analysis only.

Usage:
    conda run -n ppi python src/analysis/stability_interaction_clustering.py

Output:
    results/stability_interaction/clustering_kmeans.png
    results/stability_interaction/clustering_gmm.png
    results/stability_interaction/clustering_summary.tsv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse

# --- repo-relative path resolution (see src/paths.py) ---
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, HOME_DIR, REPO_ROOT
from utils import mutations  # noqa: E402


_PUB = REPO_ROOT
_HOME = HOME_DIR
_DB   = _PUB / "results" / "variant_dbs_all_data"
_STAB = _PUB / "results" / "variant_dbs_stability"
_OUT  = _PUB / "results" / "stability_interaction"

from analysis.stability_interaction_scatter import load_tsv_grouped, aggregate_per_variant



COSMIC_MIN_RECURRENCE = 32
K_RANGE = range(2, 7)
RANDOM_SEED = 42

SOURCE_COLORS = {
    "ClinVar Pathogenic": "#D32F2F",
    "ClinVar Benign":     "#1976D2",
    "HGMD":               "#E74C3C",
    "COSMIC recurrent":   "#B71C1C",
}


def build_pooled_data() -> pd.DataFrame:
    db_grouped = {}
    for db in ["clinvar", "hgmd", "cosmic"]:
        pred_tsv = _DB / f"{db}_mutpred_ppi_predictions.tsv"
        stab_tsv = _STAB / f"{db}_stability_predictions.tsv"
        db_grouped[db] = load_tsv_grouped(pred_tsv, stab_tsv)
        print(f"  Loaded {db}: {len(db_grouped[db]):,} unique (uniprot, variant) pairs", flush=True)

    frames = []

    with open(ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl", "rb") as f:
        pathogenic_subset = pickle.load(f)
    df = aggregate_per_variant(db_grouped["clinvar"], pathogenic_subset)
    df["source"] = "ClinVar Pathogenic"
    frames.append(df)
    print(f"  ClinVar Pathogenic: {len(df):,} variants", flush=True)

    with open(ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl", "rb") as f:
        benign_subset = pickle.load(f)
    df = aggregate_per_variant(db_grouped["clinvar"], benign_subset)
    df["source"] = "ClinVar Benign"
    frames.append(df)
    print(f"  ClinVar Benign: {len(df):,} variants", flush=True)

    with open(ANNOTATIONS_LICENSED_DIR / "hgmd_variant_subset.pkl", "rb") as f:
        hgmd_subset = pickle.load(f)
    df = aggregate_per_variant(db_grouped["hgmd"], hgmd_subset)
    df["source"] = "HGMD"
    frames.append(df)
    print(f"  HGMD: {len(df):,} variants", flush=True)

    cosmic_rec_file = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"
    with open(cosmic_rec_file, "rb") as f:
        vt_to_sites = pickle.load(f)
    cosmic_high_rec = set()
    for key, sites in vt_to_sites.items():
        if len(sites) >= COSMIC_MIN_RECURRENCE:
            parts = key.split(" ", 1)
            if len(parts) == 2:
                u, v1b = parts
                try:
                    var0 = mutations.to_zero_based(v1b)
                except ValueError:
                    continue
                cosmic_high_rec.add((u, var0))
    cosmic_all = aggregate_per_variant(db_grouped["cosmic"], subset=None)
    mask = cosmic_all.apply(lambda r: (r["uniprot"], r["variant"]) in cosmic_high_rec, axis=1)
    df = cosmic_all[mask].reset_index(drop=True)
    df["source"] = "COSMIC recurrent"
    frames.append(df)
    print(f"  COSMIC recurrent (≥{COSMIC_MIN_RECURRENCE}): {len(df):,} variants", flush=True)

    return pd.concat(frames, ignore_index=True)


def standardize(pooled: pd.DataFrame) -> np.ndarray:
    x = pooled[["mean_ddg", "max_score"]].values.astype(float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std


def select_k_kmeans(x: np.ndarray) -> tuple[int, KMeans, float]:
    best_k, best_model, best_score = None, None, -np.inf
    for k in K_RANGE:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_SEED)
        labels = km.fit_predict(x)
        score = silhouette_score(x, labels)
        print(f"  KMeans k={k}: silhouette={score:.4f}", flush=True)
        if score > best_score:
            best_k, best_model, best_score = k, km, score
    return best_k, best_model, best_score


def select_k_gmm(x: np.ndarray) -> tuple[int, GaussianMixture, float]:
    best_k, best_model, best_bic = None, None, np.inf
    for k in K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_SEED)
        gmm.fit(x)
        bic = gmm.bic(x)
        print(f"  GMM k={k}: BIC={bic:.1f}", flush=True)
        if bic < best_bic:
            best_k, best_model, best_bic = k, gmm, bic
    return best_k, best_model, best_bic


def plot_clusters(pooled: pd.DataFrame, labels: np.ndarray, title: str, out: Path,
                   gmm: GaussianMixture | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    n_clusters = len(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(pooled["mean_ddg"][mask], pooled["max_score"][mask],
                   s=4, alpha=0.3, color=cmap(c % 10), label=f"Cluster {c} (n={mask.sum():,})",
                   rasterized=True)
    if gmm is not None:
        x_std = standardize(pooled)
        x_mean = pooled[["mean_ddg", "max_score"]].values.mean(axis=0)
        x_scale = pooled[["mean_ddg", "max_score"]].values.std(axis=0)
        for c in range(n_clusters):
            mean = gmm.means_[c] * x_scale + x_mean
            cov = gmm.covariances_[c] * np.outer(x_scale, x_scale)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * np.sqrt(vals)
            ell = Ellipse(mean, width, height, angle=angle, facecolor="none",
                          edgecolor=cmap(c % 10), linewidth=2)
            ax.add_patch(ell)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean ΔΔG across partners (kcal/mol)")
    ax.set_ylabel("Max interaction disruption score")
    ax.set_title(f"{title} — colored by cluster")
    ax.legend(fontsize=8, markerscale=4)

    ax = axes[1]
    for source, color in SOURCE_COLORS.items():
        mask = pooled["source"] == source
        ax.scatter(pooled["mean_ddg"][mask], pooled["max_score"][mask],
                   s=4, alpha=0.15, color=color, label=f"{source} (n={mask.sum():,})",
                   rasterized=True)
    ax.axhline(0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.axvline(0.0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean ΔΔG across partners (kcal/mol)")
    ax.set_ylabel("Max interaction disruption score")
    ax.set_title(f"{title} — colored by source database")
    ax.legend(fontsize=8, markerscale=4)

    plt.tight_layout()
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}", flush=True)


def summarize_clusters(pooled: pd.DataFrame, labels: np.ndarray, method: str) -> list[dict]:
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        sub = pooled[mask]
        row = {
            "method": method,
            "cluster": c,
            "n": int(mask.sum()),
            "median_mean_ddg": float(sub["mean_ddg"].median()),
            "median_max_score": float(sub["max_score"].median()),
        }
        for source in SOURCE_COLORS:
            row[f"pct_{source}"] = float((sub["source"] == source).mean() * 100)
        rows.append(row)
    return rows


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    print("Building pooled dataset...", flush=True)
    pooled = build_pooled_data()
    print(f"\nTotal pooled variants: {len(pooled):,}", flush=True)

    x = standardize(pooled)

    print("\nSelecting k for KMeans (silhouette score)...", flush=True)
    k_km, km_model, km_score = select_k_kmeans(x)
    print(f"  Selected k={k_km} (silhouette={km_score:.4f})", flush=True)
    km_labels = km_model.predict(x)
    plot_clusters(pooled, km_labels, f"K-means++ (k={k_km})",
                  _OUT / "clustering_kmeans.png")

    print("\nSelecting k for GMM (BIC)...", flush=True)
    k_gmm, gmm_model, gmm_bic = select_k_gmm(x)
    print(f"  Selected k={k_gmm} (BIC={gmm_bic:.1f})", flush=True)
    gmm_labels = gmm_model.predict(x)
    plot_clusters(pooled, gmm_labels, f"Gaussian Mixture (k={k_gmm})",
                  _OUT / "clustering_gmm.png", gmm=gmm_model)

    rows = summarize_clusters(pooled, km_labels, f"kmeans_k{k_km}")
    rows += summarize_clusters(pooled, gmm_labels, f"gmm_k{k_gmm}")
    summary = pd.DataFrame(rows)
    summary.to_csv(_OUT / "clustering_summary.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"\nSaved summary → {_OUT / 'clustering_summary.tsv'}", flush=True)

    print("\nCluster summary (KMeans):")
    for r in rows:
        if r["method"] != f"kmeans_k{k_km}":
            continue
        print(f"  Cluster {r['cluster']}: n={r['n']:,}  "
              f"med_ddg={r['median_mean_ddg']:.3f}  med_score={r['median_max_score']:.3f}")

    # Directional-hypothesis check: any cluster with high ddg but low score?
    print("\nHypothesis check (stability disruption -> interaction disruption, not vice versa):")
    for r in rows:
        if r["method"] != f"kmeans_k{k_km}":
            continue
        high_ddg_low_score = r["median_mean_ddg"] > 0.5 and r["median_max_score"] < 0.5
        high_score_low_ddg = r["median_max_score"] > 0.5 and r["median_mean_ddg"] < 0.5
        if high_ddg_low_score:
            print(f"  Cluster {r['cluster']}: HIGH ΔΔG + LOW score — "
                  "REFUTES hypothesis (stability disruption without interaction disruption)")
        if high_score_low_ddg:
            print(f"  Cluster {r['cluster']}: HIGH score + LOW ΔΔG — "
                  "SUPPORTS hypothesis (interaction-only disruption)")


if __name__ == "__main__":
    main()
