#!/usr/bin/env python3
"""Stability disruption vs interaction disruption, as a 2-D enrichment per sample.

Each variant sample (ClinVar Pathogenic, COSMIC at a given recurrence, a gnomAD
allele-frequency bin, ...) becomes ONE point:

    x  stability-disruption enrichment, from mean DDG across partners per variant
    y  PPI-disruption enrichment, from max MutPred-PPI score across partners

Both axes use the same enrichment as Fig 5
(`variant_db_charts.calc_enrichment`), against gnomAD as the background:

    enrichment(f_obs) = (f_obs - f_gnomad) / (f_obs + f_gnomad)

where `f` is the fraction of variants in the sample above the threshold (0.5 for
both the interaction score and DDG, kcal/mol). The quadrants then separate
mechanism: upper-left is interaction disruption WITHOUT destabilisation, lower-
right is destabilisation without interaction loss, and lower-left is neither.

Sample membership comes from the CANONICAL per-stratum tables written by
`classify_variant_dbs.py` into `results/variant_dbs_all_data/{db}/{stratum}.csv.gz`
(columns `uniprot,variant,partner,score,n_biogrid_partners`) -- the same tables
Fig 5 is built from, so sample sizes agree with it by construction rather than
being re-derived from the restricted subset pickles. Those tables already carry
the model score, so only DDG is joined in, from the stability TSVs.

Both the stratum tables and the stability TSVs are 1-BASED, so the join needs no
base conversion; `tests/test_stability_scatter_panels.py` asserts this.

Two sample sets:

  --panels full      (default) every stratum classify_variant_dbs.py produces:
                     ClinVar x6, gnomAD across its allele-frequency bins, HGMD x3,
                     COSMIC / COSMIC-oncogene / COSMIC-TSG each across the same
                     1-2-4-8-16-32 recurrence progression, plus NDD case/control
                     and ASD.
  --panels vignette  the six disease groups plus the gnomAD gradient.

COSMIC and HGMD strata need `datasets/annotations_licensed/`. When it is absent
the affected samples are skipped with a warning and the rest are drawn.

Usage:
    conda run -n ppi python src/analysis/stability_interaction_scatter.py
    conda run -n ppi python src/analysis/stability_interaction_scatter.py --panels vignette

Output (results/stability_interaction/):
    stability_interaction_scatter.png   the 2-D enrichment scatter
    scatter_per_variant_kde.png         per-variant density behind each sample
    per_variant_summary.tsv             one row per sample: fractions, enrichments, n
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analysis import plot_style
from analysis.plot_style import SAVE_DPI
plot_style.apply()   # shared rcParams + Agg backend

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from scipy.stats import gaussian_kde     # noqa: E402

from paths import REPO_ROOT, STABILITY_INTERACTION_DIR, VARIANT_DBS_DIR, VARIANT_DBS_STABILITY_DIR  # noqa: E402

_PUB  = REPO_ROOT
# Must match the all-data model used for Fig 5 (weights/MutPred-PPI.pt).
# Default is the published all-data tree. The demonstration tier writes to
# results/variant_dbs_sahni_fragoza/ and is selected with --data-dir.
_DB   = VARIANT_DBS_DIR
_STAB = VARIANT_DBS_STABILITY_DIR
_OUT  = STABILITY_INTERACTION_DIR

# Strata that exist only when the licensed COSMIC/HGMD annotations are present.
_RESTRICTED_DBS = {"cosmic", "hgmd"}

# A variant counts as disrupted above these. 0.5 is the MutPred-PPI operating
# point used throughout the paper; 0.5 kcal/mol is the DDG cutoff Fig 5 uses.
INT_THRESHOLD = 0.5
DDG_THRESHOLD = 0.5

# Enrichment is measured against gnomAD as a whole -- the population background,
# not a curated "benign" set.
BASELINE = ("gnomad", "gnomad")

# Confidence intervals use the same resampling as Fig 5/S8/S9: a multinomial
# draw over the observed category counts, which is equivalent to a bootstrap
# over variants but costs O(n_bootstrap) instead of O(n * n_bootstrap).
N_BOOTSTRAP = 100_000
CI = (2.5, 97.5)
RANDOM_SEED = 42

# COSMIC recurrence bins, as produced by classify_variant_dbs.COSMIC_RECURRENCE_BINS.
_RECURRENCE = ["single", "2+", "4+", "8+", "16+", "32+"]
# gnomAD GroupMax allele-frequency bins, as produced by classify_variant_dbs.
# These are EXCLUSIVE ranges (prev < AF <= hi), not cumulative ceilings -- which
# is why their sizes are not monotonic in the threshold. Labelling them "AF <= x"
# would misread the figure.
_AF_BINS = ["1e-06", "1e-05", "0.0001", "0.001", "0.01", "0.1"]
# The main figure's palette keys for the allele-frequency bins. Its last bin is
# open-ended ("1e-2 < AF"), so the closed form used elsewhere misses the lookup.
_AF_PALETTE_KEYS = ["AF \u2264 1e-6", "1e-6 < AF \u2264 1e-5", "1e-5 < AF \u2264 1e-4",
                    "1e-4 < AF \u2264 1e-3", "1e-3 < AF \u2264 1e-2", "1e-2 < AF"]

_AF_LABELS = {
    "1e-06": "AF ≤ 1e-6",
    "1e-05": "1e-6 < AF ≤ 1e-5",
    "0.0001": "1e-5 < AF ≤ 1e-4",
    "0.001": "1e-4 < AF ≤ 1e-3",
    "0.01": "1e-3 < AF ≤ 1e-2",
    "0.1": "1e-2 < AF ≤ 0.1",
}


# Panel B: a fixed twelve-panel reading order, chosen so the comparison runs
# clinical -> population -> somatic -> inheritance mode, rather than following
# the recurrence/frequency progressions panel A lays out along its x-axis.
KDE_PANELS = [
    ("ClinVar pathogenic",   "clinvar", "pathogenic"),
    ("ClinVar benign",       "clinvar", "benign"),
    ("ClinVar VUS",          "clinvar", "vus"),
    ("HGMD",                 "hgmd",    "hgmd"),
    ("gnomAD",               "gnomad",  "gnomad"),
    ("COSMIC recurrent (\u226532)", "cosmic", "cosmic_32+"),
    ("Oncogene (\u22658)",   "cosmic",  "cosmic_onco_8+"),
    ("TSG (\u22658)",        "cosmic",  "cosmic_tsg_8+"),
    ("ClinVar AR",           "clinvar", "ar_pathogenic"),
    ("ClinVar AD",           "clinvar", "ad_pathogenic"),
    ("HGMD AR",              "hgmd",    "ar_hgmd"),
    ("HGMD AD",              "hgmd",    "ad_hgmd"),
]
KDE_NCOLS = 4

# Panel A mirrors the main enrichment figure exactly: the same databases in the
# same order, the same subgroup labels, the same colours. Only the two measured
# categories differ -- there it is Quasi-Null and Edgetic, here it is PPI
# disruption and stability disruption.
#
# (database, [(subgroup label, stratum file)]) in x-axis order.
ENRICHMENT_GROUPS = [
    ("ClinVar", "clinvar", [
        ("Rare Benign",   "rare_benign"),
        ("Benign",        "benign"),
        ("Pathogenic",    "pathogenic"),
        ("VUS",           "vus"),
        ("Pathogenic AR", "ar_pathogenic"),
        ("Pathogenic AD", "ad_pathogenic"),
    ]),
    ("COSMIC", "cosmic", [(lbl, f"cosmic_{st}") for lbl, st in
        [("Single", "single"), ("\u22652", "2+"), ("\u22654", "4+"),
         ("\u22658", "8+"), ("\u226516", "16+"), ("\u226532", "32+")]]),
    ("COSMIC (Onco)", "cosmic", [(lbl, f"cosmic_onco_{st}") for lbl, st in
        [("Single", "single"), ("\u22652", "2+"), ("\u22654", "4+"),
         ("\u22658", "8+"), ("\u226516", "16+"), ("\u226532", "32+")]]),
    ("COSMIC (TSG)", "cosmic", [(lbl, f"cosmic_tsg_{st}") for lbl, st in
        [("Single", "single"), ("\u22652", "2+"), ("\u22654", "4+"),
         ("\u22658", "8+"), ("\u226516", "16+"), ("\u226532", "32+")]]),
    ("HGMD", "hgmd", [("HGMD", "hgmd"), ("AR", "ar_hgmd"), ("AD", "ad_hgmd")]),
    # Labels are the MAIN FIGURE'S canonical spellings, because the shared
    # palette is keyed on them. `_display_label` lowercases Case/Control for the
    # tick text exactly as the main figure does -- looking the colour up with
    # the lowercased form silently returns the default grey.
    ("gnomAD", "gnomad", list(zip(_AF_PALETTE_KEYS,
                                  [f"gnomad_upper_af_{b}" for b in _AF_BINS]))),
    ("NDD", "neurodev", [("NDD Case", "ndd_case"), ("NDD Control", "ndd_control")]),
    ("ASD", "asd", [("ASD Case", "asd")]),
]

# Colours come from the main enrichment figure, so a subgroup keeps its colour
# between the two. `_get_enrich_color` keys oncogene/TSG separately from plain
# COSMIC, which is why the stratum name has to be inspected rather than the db.
# Panel C uses ONE colour scheme throughout. Colouring each density by its
# sample invited the eye to compare hues across panels, which carries no meaning
# here -- the comparison is of distribution SHAPE against a common pair of axes.
KDE_CMAP = "Reds"
KDE_LINE = "#B71C1C"


def _get_enrich_color(db: str, stratum: str, label: str) -> str:
    """Delegate to the main figure's palette, keyed the way it expects."""
    from analysis.variant_db_charts import _get_enrich_color as fig5_color
    key = ("cosmic_onco" if stratum.startswith("cosmic_onco")
           else "cosmic_tsg" if stratum.startswith("cosmic_tsg")
           else "gnomad_af" if stratum.startswith("gnomad_upper_af") else db)
    return fig5_color(key, label)


def _panel_color(label: str, db: str) -> str:
    """Retained for the panel spec; panel C is drawn in one colour scheme."""
    return KDE_LINE


# ── Data ──────────────────────────────────────────────────────────────────────

def load_stability(db: str) -> pd.DataFrame | None:
    """(interactor, partner, mutation) -> ddg_kcalmol for one database.

    `mutation` is 1-based, matching the `variant` column of the stratum tables.
    """
    path = _STAB / f"{db}_stability_predictions.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype={"interactor": str, "partner": str,
                                            "mutation": str})
    missing = {"interactor", "partner", "mutation", "ddg_kcalmol"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}; regenerate it with "
                         f"src/variant_db_inference/run_stability_inference.py")
    return df.rename(columns={"interactor": "uniprot", "mutation": "variant"})


def load_panel(db: str, stratum: str, stability: pd.DataFrame) -> pd.DataFrame | None:
    """Aggregate one canonical stratum table to one row per unique variant."""
    path = _DB / db / f"{stratum}.csv.gz"
    if not path.exists():
        return None
    rows = pd.read_csv(path, dtype={"uniprot": str, "variant": str, "partner": str})

    merged = rows.merge(stability, on=["uniprot", "partner", "variant"], how="inner")
    if merged.empty:
        return merged.assign(max_score=[], mean_ddg=[])

    agg = (merged.groupby(["uniprot", "variant"], sort=False)
                 .agg(max_score=("score", "max"),
                      mean_score=("score", "mean"),
                      n_partners=("score", "size"),
                      mean_ddg=("ddg_kcalmol", "mean"),
                      max_ddg=("ddg_kcalmol", "max"))
                 .reset_index())
    return agg


def sample_fractions(df: pd.DataFrame,
                     int_threshold: float = INT_THRESHOLD,
                     ddg_threshold: float = DDG_THRESHOLD) -> tuple[float, float, int]:
    """(fraction PPI-disrupted, fraction destabilised, n variants) for one sample."""
    if df is None or df.empty:
        return 0.0, 0.0, 0
    return (float((df["max_score"] >= int_threshold).mean()),
            float((df["mean_ddg"] >= ddg_threshold).mean()),
            len(df))


def bootstrap_fractions(df: pd.DataFrame, int_threshold: float, ddg_threshold: float,
                        n_bootstrap: int, rng) -> np.ndarray:
    """(n_bootstrap, 2) replicates of (fraction PPI-disrupted, fraction destabilised).

    Resamples the 2x2 JOINT contingency of (disrupted, destabilising) rather
    than the two fractions independently. Both are measured on the same
    variants and are strongly correlated -- a variant that destabilises the fold
    often disrupts the interface too -- so independent resampling would
    misstate the uncertainty in where a sample sits in the plane.
    """
    if df is None or df.empty:
        return np.zeros((n_bootstrap, 2))
    d = (df["max_score"] >= int_threshold).to_numpy()
    g = (df["mean_ddg"] >= ddg_threshold).to_numpy()
    counts = np.array([np.sum(d & g), np.sum(d & ~g),
                       np.sum(~d & g), np.sum(~d & ~g)], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.zeros((n_bootstrap, 2))
    boot = rng.multinomial(int(total), counts / total, size=n_bootstrap) / total
    return np.column_stack([boot[:, 0] + boot[:, 1],      # PPI-disrupted
                            boot[:, 0] + boot[:, 2]])     # destabilising


def calc_enrichment(f_obs: float, f_base: float) -> float:
    """Same statistic as Fig 5 (`variant_db_charts.calc_enrichment`).

    Bounded in [-1, 1] and symmetric in the two fractions, so a sample twice as
    disrupted as background and one half as disrupted sit equally far from zero.
    """
    total = f_obs + f_base
    return (f_obs - f_base) / total if total > 0 else 0.0


def load_enrichment_samples(int_threshold, ddg_threshold, n_bootstrap):
    """Fractions + bootstrap replicates for every sample, and the gnomAD baseline.

    Returns (groups, skipped) where `groups` mirrors ENRICHMENT_GROUPS with each
    subgroup carrying its observed fractions, its bootstrap replicates and n.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    stability: dict[str, pd.DataFrame | None] = {}

    def _stab(db):
        if db not in stability:
            stability[db] = load_stability(db)
            if stability[db] is None:
                why = ("licensed annotations absent" if db in _RESTRICTED_DBS
                       else "stability TSV absent")
                print(f"[skip] {db}: {why}", flush=True)
        return stability[db]

    base_stab = _stab(BASELINE[0])
    if base_stab is None:
        raise ValueError("the gnomAD stability predictions are required as the "
                         "enrichment background")
    base_df = load_panel(BASELINE[0], BASELINE[1], base_stab)
    if base_df is None or base_df.empty:
        raise ValueError(f"the gnomAD baseline table "
                         f"({BASELINE[0]}/{BASELINE[1]}) is empty or absent")
    f_int_base, f_ddg_base, n_base = sample_fractions(base_df, int_threshold, ddg_threshold)
    base_boot = bootstrap_fractions(base_df, int_threshold, ddg_threshold,
                                    n_bootstrap, rng)
    print(f"gnomAD baseline: n={n_base:,}  {100 * f_int_base:.2f}% PPI-disrupted  "
          f"{100 * f_ddg_base:.2f}% destabilising", flush=True)
    print(f"bootstrapping x {n_bootstrap:,} replicates...", flush=True)

    groups, skipped = [], []
    for display, db, subgroups in ENRICHMENT_GROUPS:
        stab = _stab(db)
        kept = []
        for label, stratum in subgroups:
            df = load_panel(db, stratum, stab) if stab is not None else None
            if df is None or df.empty:
                skipped.append({"group": display, "label": label, "db": db,
                                "stratum": stratum,
                                "reason": "stratum table absent"
                                          if stab is not None else
                                          "no stability predictions"})
                continue
            f_int, f_ddg, n = sample_fractions(df, int_threshold, ddg_threshold)
            boot = bootstrap_fractions(df, int_threshold, ddg_threshold,
                                       n_bootstrap, rng)
            kept.append({"label": label, "db": db, "stratum": stratum, "df": df,
                         "n": n, "f_int": f_int, "f_ddg": f_ddg, "boot": boot,
                         "color": _get_enrich_color(db, stratum, label)})
            print(f"  {display} / {label}: {n:,} variants", flush=True)
        if kept:
            groups.append({"display": display, "db": db, "subgroups": kept})
    return groups, skipped, base_boot, (f_int_base, f_ddg_base)


# ── Plotting ──────────────────────────────────────────────────────────────────

# Samples that form an ordered progression are drawn as a connected trail rather
# than as unrelated dots: the direction of travel is the point of the figure.
# (colour, label, head) -- `head` is which END of the declared stratum order the
# arrow points at, because the two progressions are declared in opposite senses:
# gnomAD bins run rarest-first, COSMIC recurrence bins run least-recurrent-first.
# Pointing the arrow at the wrong end reverses the story the figure tells.
_TRAILS = {
    "gnomAD (by allele-frequency bin)": ("#9e9e9e", "common \u2192 rare", "first"),
    "COSMIC (by recurrence)":           ("#B71C1C", "rare \u2192 recurrent", "last"),
    "COSMIC oncogenes (by recurrence)": ("#FF6D00", "rare \u2192 recurrent", "last"),
    "COSMIC tumour suppressors (by recurrence)": ("#FFA000", "rare \u2192 recurrent", "last"),
}


def _enrichment_points(drawn, int_threshold, ddg_threshold,
                       n_bootstrap: int = N_BOOTSTRAP):
    """One (x, y) per sample with bootstrap CIs, plus the fractions behind it."""
    by_key = {(p["db"], p["stratum"]): p for p in drawn}
    base = by_key.get(BASELINE)
    if base is None:
        raise ValueError(
            f"the gnomAD baseline ({BASELINE[0]}/{BASELINE[1]}) is required to "
            f"compute enrichment, and is not among the samples that loaded")
    f_int_base, f_ddg_base, _ = sample_fractions(base["df"], int_threshold, ddg_threshold)

    rng = np.random.default_rng(RANDOM_SEED)
    # The background is resampled too, and each replicate of a sample is paired
    # with the SAME replicate of the background -- otherwise the shared
    # uncertainty in the background would be counted once per sample.
    base_boot = bootstrap_fractions(base["df"], int_threshold, ddg_threshold,
                                    n_bootstrap, rng)
    print(f"bootstrapping {len(drawn)} samples x {n_bootstrap:,} replicates...",
          flush=True)

    out = []
    for p in drawn:
        f_int, f_ddg, n = sample_fractions(p["df"], int_threshold, ddg_threshold)
        boot = bootstrap_fractions(p["df"], int_threshold, ddg_threshold,
                                   n_bootstrap, rng)
        # calc_enrichment elementwise; the denominator is never 0 in practice
        # but guard anyway so an all-empty sample cannot produce a NaN band.
        def _enr(obs, bas):
            tot = obs + bas
            return np.divide(obs - bas, tot, out=np.zeros_like(tot), where=tot > 0)
        y_boot = _enr(boot[:, 0], base_boot[:, 0])
        x_boot = _enr(boot[:, 1], base_boot[:, 1])
        out.append({**p,
                    "f_int": f_int, "f_ddg": f_ddg, "n": n,
                    "x": calc_enrichment(f_ddg, f_ddg_base),
                    "y": calc_enrichment(f_int, f_int_base),
                    "x_lo": float(np.percentile(x_boot, CI[0])),
                    "x_hi": float(np.percentile(x_boot, CI[1])),
                    "y_lo": float(np.percentile(y_boot, CI[0])),
                    "y_hi": float(np.percentile(y_boot, CI[1]))})
    return out, (f_int_base, f_ddg_base)


def _place_labels(ax, points, stroke, fontsize=9.5):
    """Greedy non-overlapping label placement around a crowded scatter.

    Tries eight offsets per point and keeps the first whose rendered box misses
    every box already placed, so no label is dropped -- a collided label is
    worse than an oddly-placed one, but a missing label is worse than both.
    """
    if not points:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [(9, 9), (-9, 9), (9, -9), (-9, -11), (13, 0),
                  (-13, 0), (0, 13), (0, -14)]
    placed = []
    # Label the most extreme points first: they have the most free space and
    # are the ones a reader looks for.
    order = sorted(points, key=lambda p: -(p["x"] ** 2 + p["y"] ** 2))
    for p in order:
        best = None
        for dx, dy in candidates:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
            va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            t = ax.annotate(p["label"], (p["x"], p["y"]),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=p["color"],
                            fontweight="bold", ha=ha, va=va, zorder=6)
            bb = t.get_window_extent(renderer=renderer)
            if not any(bb.overlaps(o) for o in placed):
                best = (t, bb)
                break
            t.remove()
        if best is None:                       # every direction was taken
            dx, dy = candidates[0]
            t = ax.annotate(p["label"], (p["x"], p["y"]),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=fontsize, color=p["color"],
                            fontweight="bold", ha="left", va="bottom", zorder=6)
            best = (t, t.get_window_extent(renderer=renderer))
        best[0].set_path_effects(stroke)
        placed.append(best[1])


# The two categories panel A measures, in the order they are stacked. The main
# figure stacks Quasi-Null over Edgetic; this stacks PPI over stability.
COMPONENTS = [("Max PPI Disruption", "f_int", 0),
              ("Stability Disruption", "f_ddg", 1)]


def _enrichment_replicates(sub, base_boot, comp_idx):
    """Per-replicate enrichment of one subgroup against the paired background."""
    obs, bas = sub["boot"][:, comp_idx], base_boot[:, comp_idx]
    tot = obs + bas
    return np.divide(obs - bas, tot, out=np.zeros_like(tot), where=tot > 0)


def _display_label(label: str) -> str:
    """Tick text for a palette key -- the main figure's own transformation."""
    return label.replace("Case", "case").replace("Control", "control")


def _layout(groups):
    """x positions, tick labels and group boundaries -- Fig 5's spacing."""
    ticks, labels, bounds, x = [], [], [], 0
    for g in groups:
        start = x
        for sub in g["subgroups"]:
            ticks.append(x)
            labels.append(f"{_display_label(sub['label'])} (n={sub['n']:,})")
            x += 1
        bounds.append((start, x - 0.5, g["display"]))
        x += 1.5
    return ticks, labels, bounds, x


def draw_enrichment_layer(ax, groups, base_boot, comp_idx, comp_name,
                          n_tests, layout, is_top):
    """One layer of panel A: a bar per sample, median with 16/84 interval.

    Significance marking matches the main figure: `*` survives Bonferroni
    correction across every bar in both layers, `•` is nominally significant
    only.
    """
    ticks, labels, bounds, final_x = layout
    alpha_bonf = 0.05 / n_tests
    xp = 0
    for g in groups:
        for sub in g["subgroups"]:
            vals = _enrichment_replicates(sub, base_boot, comp_idx)
            median = float(np.median(vals))
            p16, p84 = np.percentile(vals, [16, 84])
            if median >= 0:
                sig_bonf = np.percentile(vals, 100 * alpha_bonf) > 0
                sig_uncorr = np.percentile(vals, 5) > 0
            else:
                sig_bonf = np.percentile(vals, 100 * (1 - alpha_bonf)) < 0
                sig_uncorr = np.percentile(vals, 95) < 0
            ax.bar(xp, median, width=0.8, color=sub["color"], edgecolor="black",
                   linewidth=1.2, alpha=0.9 if sig_bonf else 0.55)
            ax.errorbar(xp, median, yerr=[[median - p16], [p84 - median]],
                        fmt="none", ecolor="black", capsize=4, linewidth=2, alpha=0.7)
            if sig_bonf or sig_uncorr:
                y = (p84 + 0.05) if median > 0 else (p16 - 0.05)
                ax.text(xp, y, "*" if sig_bonf else "\u2022", ha="center",
                        va="bottom" if median > 0 else "top",
                        fontsize=20 if sig_bonf else 12,
                        fontweight="bold", color="black")
            sub[f"median_{comp_idx}"] = median
            sub[f"p16_{comp_idx}"] = float(p16)
            sub[f"p84_{comp_idx}"] = float(p84)
            sub[f"sig_{comp_idx}"] = "bonferroni" if sig_bonf else (
                "nominal" if sig_uncorr else "ns")
            xp += 1
        xp += 1.5

    # Group headings once, on the upper layer: the two layers share an x axis.
    if is_top:
        for start, end, name in bounds:
            ax.text((start + end) / 2, 0.90, name,
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                              edgecolor="gray", linewidth=1.5),
                    transform=ax.get_xaxis_transform())

    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_ylabel(f"{comp_name}\nEnrichment", fontsize=14, fontweight="bold")
    ax.set_xticks(ticks)
    ax.set_xlim(-1.5, final_x - 2 + 1.5)
    ax.set_ylim(-1, 1.05)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)
    if is_top:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
        ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.spines["top"].set_visible(False)


def _kde_limits(panels):
    ddg = np.concatenate([p["df"]["mean_ddg"].values for p in panels])
    return (float(np.percentile(ddg, 1)), float(np.percentile(ddg, 99))), (0.0, 1.0)


def draw_kde_grid(fig, gs, panels, max_n: int = 20000):
    """Panel B: per-variant density for each sample, as one flush grid.

    Panels sit edge to edge with shared axes; only the outer edges carry tick
    labels, so the grid reads as a single plot rather than twelve small ones.
    """
    ncols = KDE_NCOLS
    nrows = int(np.ceil(len(panels) / ncols))
    sub = gs.subgridspec(nrows, ncols, wspace=0.0, hspace=0.0)
    xlim, ylim = _kde_limits(panels)
    rng = np.random.default_rng(42)

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    axes = []
    base = None
    for i, p in enumerate(panels):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(sub[r, c], sharex=base, sharey=base)
        base = base or ax
        axes.append(ax)
        df = p["df"]

        n = min(max_n, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        x = df.iloc[idx]["mean_ddg"].values
        y = df.iloc[idx]["max_score"].values

        if n > 10:
            try:
                Z = gaussian_kde(np.vstack([x, y]), bw_method=0.15)(positions)
                ax.contourf(XX, YY, Z.reshape(XX.shape), levels=12,
                            cmap=KDE_CMAP, alpha=0.7)
                ax.contour(XX, YY, Z.reshape(XX.shape), levels=6,
                           colors=KDE_LINE, linewidths=0.5, alpha=0.6)
            except Exception:                                  # noqa: BLE001
                ax.scatter(x, y, c=KDE_LINE, alpha=0.05, s=2, rasterized=True)
        else:
            ax.scatter(x, y, c=KDE_LINE, alpha=0.3, s=6, rasterized=True)

        ax.axhline(INT_THRESHOLD, color="grey", lw=0.7, ls="--", alpha=0.5)
        ax.axvline(0.0, color="grey", lw=0.7, ls="--", alpha=0.5)

        med_ddg, med_score = df["mean_ddg"].median(), df["max_score"].median()
        ax.axvline(med_ddg, color=KDE_LINE, lw=1.4, ls=":", alpha=0.9,
                   label=f"median \u0394\u0394G = {med_ddg:.2f}")
        ax.axhline(med_score, color=KDE_LINE, lw=1.4, ls="-.", alpha=0.9,
                   label=f"median PPI disruption score = {med_score:.2f}")
        ax.legend(fontsize=5.6, loc="upper right", framealpha=0.85,
                  borderpad=0.25, handlelength=1.3, borderaxespad=0.2)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Title inside the panel, so panels can sit flush.
        ax.text(0.03, 0.97, f"{p['label']}\n(n={len(df):,})", transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left", fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75))

        # Only the outer edges of the whole grid are labelled.
        if c != 0:
            ax.tick_params(labelleft=False)
        if r != nrows - 1:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=7)
    return axes, nrows, ncols


def make_figure(groups, base_boot, kde_panels, out: Path,
                demo_tier: bool = False) -> None:
    """One figure: (A) the two-layer enrichment bars, (B) the density grid."""
    from matplotlib.gridspec import GridSpec

    layout = _layout(groups)
    n_samples = sum(len(g["subgroups"]) for g in groups)
    n_tests = n_samples * len(COMPONENTS)
    print(f"  Bonferroni n_tests = {n_tests}", flush=True)

    width = max(min(18 * (layout[3] / 35.0), 18), 10)
    fig = plt.figure(figsize=(width, 22))
    # Two layers of panel A are flush with each other (they share an x axis);
    # panel B needs clear air beneath A's rotated tick labels.
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.15], hspace=0.30)
    gs_a = outer[0].subgridspec(2, 1, hspace=0.0)

    axes_a = [fig.add_subplot(gs_a[0]), fig.add_subplot(gs_a[1])]
    for i, ((name, _key, comp_idx), ax) in enumerate(zip(COMPONENTS, axes_a)):
        draw_enrichment_layer(ax, groups, base_boot, comp_idx, name,
                              n_tests, layout, is_top=(i == 0))

    axes, nrows, ncols = draw_kde_grid(fig, outer[1], kde_panels)

    # Panel letters share one x in FIGURE coordinates. Placing each in its own
    # axes coordinates puts them at different absolute positions, because the
    # enrichment layers and the density grid have different left margins.
    fig.canvas.draw()
    letter_x = min(ax.get_position().x0 for ax in (*axes_a, *axes)) - 0.045
    for letter, ax in zip("ABC", (*axes_a, axes[0])):
        fig.text(letter_x, ax.get_position().y1, f"({letter})",
                 fontsize=20, fontweight="bold", va="bottom", ha="left")

    left = axes[0].get_position().x0
    right = axes[min(ncols, len(axes)) - 1].get_position().x1
    bottom = axes[-1].get_position().y0
    top = axes[0].get_position().y1
    fig.text((left + right) / 2, bottom - 0.021,
             "Mean \u0394\u0394G across partners (kcal/mol)",
             ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(left - 0.042, (bottom + top) / 2,
             "Max PPI disruption score across partners",
             ha="right", va="center", rotation=90, fontsize=12, fontweight="bold")

    if demo_tier:
        plot_style.demo_stamp(fig)
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="classified stratum tables (default: "
                         "results/variant_dbs_all_data)")
    ap.add_argument("--demo-tier", action="store_true",
                    help="stamp the figure as coming from the Sahni+Fragoza "
                         "demonstration model (see run_variant_db_inference.py "
                         "--model-tier)")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                    help=f"bootstrap replicates for panel A (default "
                         f"{N_BOOTSTRAP:,}, matching the main enrichment figure)")
    ap.add_argument("--int-threshold", type=float, default=INT_THRESHOLD,
                    help=f"MutPred-PPI score at or above which a variant counts "
                         f"as PPI-disrupting (default {INT_THRESHOLD})")
    ap.add_argument("--ddg-threshold", type=float, default=DDG_THRESHOLD,
                    help=f"DDG (kcal/mol) at or above which a variant counts as "
                         f"destabilising (default {DDG_THRESHOLD})")
    args = ap.parse_args()

    if args.data_dir:
        global _DB
        _DB = Path(args.data_dir)

    _OUT.mkdir(parents=True, exist_ok=True)
    groups, skipped, base_boot, (f_int_base, f_ddg_base) = load_enrichment_samples(
        args.int_threshold, args.ddg_threshold, args.n_bootstrap)

    if not groups:
        print("No sample has both a stratum table and stability predictions. "
              "Run classify_variant_dbs.py and run_stability_inference.py first.")
        return 1
    if skipped:
        print(f"\n{len(skipped)} sample(s) skipped:", flush=True)
        for s in skipped:
            print(f"  [skip] {s['label']} ({s['db']}/{s['stratum']}): {s['reason']}")

    # Panel B: a fixed reading order, independent of panel A's grouping.
    kde_panels, kde_missing = [], []
    stab_cache: dict[str, object] = {}
    for label, db, stratum in KDE_PANELS:
        if db not in stab_cache:
            stab_cache[db] = load_stability(db)
        stab = stab_cache[db]
        df = load_panel(db, stratum, stab) if stab is not None else None
        if df is None or df.empty:
            kde_missing.append(f"{label} ({db}/{stratum})")
            continue
        kde_panels.append({"label": label, "db": db, "stratum": stratum,
                           "df": df, "color": _panel_color(label, db)})
    if kde_missing:
        print(f"\npanel B: {len(kde_missing)} of {len(KDE_PANELS)} samples "
              f"unavailable: {', '.join(kde_missing)}", flush=True)
    if not kde_panels:
        print("panel B has no samples; cannot draw the figure.")
        return 1

    out = _OUT / "stability_interaction_scatter.png"
    make_figure(groups, base_boot, kde_panels, out, demo_tier=args.demo_tier)

    # draw_enrichment_layer records the per-bar statistics as it draws them.
    rows = []
    for g in groups:
        for sub in g["subgroups"]:
            rows.append({
                "group": g["display"], "sample": sub["label"],
                "db": sub["db"], "stratum": sub["stratum"], "n_variants": sub["n"],
                "pct_ppi_disrupted": 100 * sub["f_int"],
                "pct_destabilising": 100 * sub["f_ddg"],
                "ppi_enrichment": sub.get("median_0"),
                "ppi_enrichment_p16": sub.get("p16_0"),
                "ppi_enrichment_p84": sub.get("p84_0"),
                "ppi_significance": sub.get("sig_0"),
                "stability_enrichment": sub.get("median_1"),
                "stability_enrichment_p16": sub.get("p16_1"),
                "stability_enrichment_p84": sub.get("p84_1"),
                "stability_significance": sub.get("sig_1"),
                "status": "drawn",
            })
    rows += [{"group": s["group"], "sample": s["label"], "db": s["db"],
              "stratum": s["stratum"], "n_variants": 0, "status": s["reason"]}
             for s in skipped]
    pd.DataFrame(rows).to_csv(_OUT / "per_variant_summary.tsv", sep="\t",
                              index=False, float_format="%.4f")

    n = sum(len(g["subgroups"]) for g in groups)
    print(f"\nA: {n} samples in {len(groups)} groups   "
          f"B: {len(kde_panels)} panels   ({len(skipped)} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
