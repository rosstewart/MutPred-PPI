#!/usr/bin/env python
"""Generate publication-quality charts for variant database analysis.

Produces four groups of figures from pre-computed model scores and
edgotype class arrays:

  1. Score-distribution histograms (--score-histograms)
     Reads {data_dir}/{dataset}/{subset}_posts.npy per configured group.

  2. Uncontrolled edgotype enrichment bootstrap (--edgotype-bootstrap)
     Reads {data_dir}/{dataset}/{subset}.csv.gz and derives edgotypes.
     Runs bootstrap sampling of full edgotype arrays, then plots
     Quasi-Null and Edgetic enrichment relative to gnomAD baseline.

  3. Partner-count-controlled edgotype bootstrap (--controlled-bootstrap)
     Reads {data_dir}/{dataset}/{subset}.csv.gz for per-variant score lists.
     Samples exactly k partners per variant for each bootstrap iteration.
     Runs for k = 3, 5, 7 (or just k = 3 with --k3-only).

  4. COSMIC tumor-site variation analysis (--tumor-sites)
     Reads tumor_site_to_edgotypes pickles, runs permutation test, saves
     permutation histogram and coefficient-of-variation comparison.

All outputs go to --output-dir (default: same directory as --data-dir).

Usage:
    python variant_db_charts.py \
        --data-dir $MUTPRED_REPO/results/variant_dbs_sufficient_partners \
        --output-dir $MUTPRED_REPO/results/variant_dbs_sufficient_partners \
        --n-bootstrap 100000 \
        --score-histograms \
        --edgotype-bootstrap \
        --controlled-bootstrap \
        --tumor-sites
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import matplotlib
from analysis import edgotypes, plot_style
from analysis import plot_style
from analysis.plot_style import SAVE_DPI


# Set by --demo-tier: every figure this run writes is stamped as coming from the
# Sahni+Fragoza demonstration model rather than the published all-data model.
_DEMO_STAMP = False


def _save(out) -> None:
    """Save the current figure, stamping it when running a non-published tier."""
    if _DEMO_STAMP:
        plot_style.demo_stamp(plt.gcf())
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches="tight")
plot_style.apply()   # shared rcParams + Agg backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredText
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

# rcParams now come from analysis.plot_style.apply(), so every figure in the
# paper shares one font size, axis weight and output resolution. This block
# was duplicated verbatim in two scripts and absent from the other thirteen.

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

dataset_colors = {
    "hgmd": ("#E74C3C", "darkred"),
    "gnomad": ("#4CAF50", "#2E7D32"),
    "gnomad_not_disease_gene": ("#4CAF50", "#2E7D32"),
    "gnomad_disease_gene": ("#D32F2F", "#B71C1C"),
    "gnomad_upper_af_1e-06": ("#FF7043", "#F4511E"),
    "gnomad_upper_af_1e-05": ("#FFA726", "#FF9100"),
    "gnomad_upper_af_0.0001": ("#FFCA28", "#FFB300"),
    "gnomad_upper_af_0.001": ("#9CCC65", "#7CB342"),
    "gnomad_upper_af_0.01": ("#66BB6A", "#4CAF50"),
    "gnomad_upper_af_0.1": ("#4CAF50", "#2E7D32"),
    "ndd_case": ("#9C27B0", "#6A1B9A"),
    "ndd_control": ("#607D8B", "#37474F"),
    "asd": ("#FF9800", "#E65100"),
    "pathogenic": ("#D32F2F", "#B71C1C"),
    "benign": ("#1976D2", "#0D47A1"),
    "vus": ("#9E9E9E", "#616161"),
    "cosmic_single": ("#FDD835", "#F9A825"),
    "cosmic_32+": ("#B71C1C", "#7F0000"),
    # extended COSMIC subsets
    "cosmic_2+": ("#FFCA28", "#FFB300"),
    "cosmic_4+": ("#FF8F00", "#FF6F00"),
    "cosmic_8+": ("#FF5722", "#F4511E"),
    "cosmic_16+": ("#E64A19", "#D84315"),
    # oncogenes
    "cosmic_onco_single": ("#FFE0B2", "#FFCC80"),
    "cosmic_onco_2+": ("#FFAB40", "#FF9100"),
    "cosmic_onco_4+": ("#FF6D00", "#FF5722"),
    "cosmic_onco_8+": ("#FF3D00", "#DD2C00"),
    "cosmic_onco_16+": ("#BF360C", "#A52714"),
    "cosmic_onco_32+": ("#7E2010", "#5D1510"),
    # tumour suppressors
    "cosmic_tsg_single": ("#FFFDE7", "#FFF176"),
    "cosmic_tsg_2+": ("#FFD54F", "#FFC107"),
    "cosmic_tsg_4+": ("#FFA000", "#FF8F00"),
    "cosmic_tsg_8+": ("#FF6E40", "#FF5722"),
    "cosmic_tsg_16+": ("#EF5350", "#E53935"),
    "cosmic_tsg_32+": ("#C62828", "#B71C1C"),
}

dataset_labels = {
    "hgmd": "HGMD",
    "gnomad": "gnomAD",
    "gnomad_not_disease_gene": "Not Disease Gene",
    "gnomad_disease_gene": "Disease Gene",
    "gnomad_upper_af_1e-06": "AF ≤ 1e-6",
    "gnomad_upper_af_1e-05": "1e-6 < AF ≤ 1e-5",
    "gnomad_upper_af_0.0001": "1e-5 < AF ≤ 1e-4",
    "gnomad_upper_af_0.001": "1e-4 < AF ≤ 1e-3",
    "gnomad_upper_af_0.01": "1e-3 < AF ≤ 1e-2",
    "gnomad_upper_af_0.1": "1e-2 < AF",
    "neurodev": "Neurodevelopmental Disorder",
    "ndd_case": "Case",
    "ndd_control": "Control",
    "asd": "ASD Case",
    "clinvar": "ClinVar",
    "pathogenic": "Pathogenic",
    "benign": "Benign",
    "vus": "VUS",
    "cosmic": "COSMIC Cancer-Linked",
    "cosmic_single": "Single Occurrence",
    "cosmic_2+": "Recurrence ≥ 2",
    "cosmic_4+": "Recurrence ≥ 4",
    "cosmic_8+": "Recurrence ≥ 8",
    "cosmic_16+": "Recurrence ≥ 16",
    "cosmic_32+": "Recurrence ≥ 32",
    "cosmic_onco_single": "Single Occurrence",
    "cosmic_onco_2+": "Recurrence ≥ 2",
    "cosmic_onco_4+": "Recurrence ≥ 4",
    "cosmic_onco_8+": "Recurrence ≥ 8",
    "cosmic_onco_16+": "Recurrence ≥ 16",
    "cosmic_onco_32+": "Recurrence ≥ 32",
    "cosmic_tsg_single": "Single Occurrence",
    "cosmic_tsg_2+": "Recurrence ≥ 2",
    "cosmic_tsg_4+": "Recurrence ≥ 4",
    "cosmic_tsg_8+": "Recurrence ≥ 8",
    "cosmic_tsg_16+": "Recurrence ≥ 16",
    "cosmic_tsg_32+": "Recurrence ≥ 32",
    "ar_pathogenic": "AR",
    "ad_pathogenic": "AD",
    "ar_hgmd": "AR",
    "ad_hgmd": "AD",
}

# dataset_enrichment_labels: short display labels for enrichment plot x-axis
dataset_enrichment_labels = {
    "hgmd": ["HGMD", "AR", "AD"],
    "gnomad_af": ["AF ≤ 1e-6", "1e-6 < AF ≤ 1e-5", "1e-5 < AF ≤ 1e-4",
                  "1e-4 < AF ≤ 1e-3", "1e-3 < AF ≤ 1e-2", "1e-2 < AF"],
    "asd": "ASD Case",
    "neurodev": ["NDD Case", "NDD Control"],
    "clinvar": ["Rare Benign", "Benign", "Pathogenic", "VUS", "Pathogenic AR", "Pathogenic AD"],
    "cosmic": ["Single", "≥2", "≥4", "≥8", "≥16", "≥32"],
    "cosmic_onco": ["Single", "≥2", "≥4", "≥8", "≥16", "≥32"],
    "cosmic_tsg": ["Single", "≥2", "≥4", "≥8", "≥16", "≥32"],
}

# Configurations: (base_dataset, [datasets], stats_loc, save_name, alpha, n_partners_suffix)
CONFIGURATIONS = [
    ("hgmd", ["hgmd", "ar_hgmd", "ad_hgmd"], "upper right", "hgmd", None, ""),
    ("gnomad", ["gnomad"], "upper right", "gnomad", None, ""),
    ("gnomad", ["gnomad_disease_gene", "gnomad_not_disease_gene"],
     "upper right", "gnomad_disease_gene", None, ""),
    ("gnomad", ["gnomad_upper_af_1e-06", "gnomad_upper_af_1e-05",
                "gnomad_upper_af_0.0001", "gnomad_upper_af_0.001",
                "gnomad_upper_af_0.01", "gnomad_upper_af_0.1"],
     "upper right", "gnomad_af", None, "_af"),
    ("asd", ["asd"], "upper right", "asd", None, ""),
    ("neurodev", ["ndd_case", "ndd_control"], "upper right", "neurodev", None, ""),
    ("clinvar", ["rare_benign", "benign", "pathogenic", "vus", "ar_pathogenic", "ad_pathogenic"],
     "upper right", "clinvar", None, ""),
    ("cosmic", ["cosmic_single", "cosmic_2+", "cosmic_4+", "cosmic_8+",
                "cosmic_16+", "cosmic_32+"], "upper right", "cosmic", None, ""),
    ("cosmic", ["cosmic_onco_single", "cosmic_onco_2+", "cosmic_onco_4+",
                "cosmic_onco_8+", "cosmic_onco_16+", "cosmic_onco_32+"],
     "upper right", "cosmic_onco", None, "_onco"),
    ("cosmic", ["cosmic_tsg_single", "cosmic_tsg_2+", "cosmic_tsg_4+",
                "cosmic_tsg_8+", "cosmic_tsg_16+", "cosmic_tsg_32+"],
     "upper right", "cosmic_tsg", None, "_tsg"),
]

HISTOGRAM_CONFIGS = [
    ("hgmd", ["hgmd"], "upper left", "hgmd", None),
    ("gnomad", ["gnomad"], "upper right", "gnomad", None),
    ("gnomad", ["gnomad_disease_gene", "gnomad_not_disease_gene"],
     "upper right", "gnomad_disease_gene", None),
    ("asd", ["asd"], "upper left", "asd", None),
    ("neurodev", ["ndd_case", "ndd_control"], "upper left", "neurodev", None),
    ("clinvar", ["rare_benign", "benign", "pathogenic", "vus"], "upper right", "clinvar", 0.5),
    ("cosmic", ["cosmic_single", "cosmic_32+"], "upper right", "cosmic", 0.5),
]


# ---------------------------------------------------------------------------
# Score histogram
# ---------------------------------------------------------------------------

def plot_multi_class_histogram(base_dataset_name, datasets, data_base_dir,
                               save_name=None, stats_loc="best",
                               title_suffix="Variant-Partner Combinations",
                               alpha=None, output_dir=None):
    if alpha is None:
        alpha = 0.7 if len(datasets) == 1 else 0.6

    fig, ax = plt.subplots(figsize=(8, 6))
    all_data, all_labels = [], []

    for dataset in datasets:
        data_path = f"{data_base_dir}/{base_dataset_name}/{dataset}_posts.npy"
        if not os.path.exists(data_path):
            print(f"  Warning: missing {data_path}")
            continue
        data = np.load(data_path)
        all_data.append(data)
        all_labels.append(dataset_labels.get(dataset, dataset))
        ax.hist(data, bins=25 if len(data) < 10_000 else 50,
                density=True, alpha=alpha,
                color=dataset_colors[dataset][0],
                edgecolor=dataset_colors[dataset][1],
                linewidth=0.8,
                label=dataset_labels.get(dataset, dataset))

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Interaction Loss Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    if "gnomad_disease_gene" in datasets:
        title_suffix += ":\nDisease Gene Comparison"

    if len(datasets) == 1:
        title = f"Distribution of Interaction Loss Scores\nfor {all_labels[0]} {title_suffix}"
    else:
        title = (f"Distribution of Interaction Loss Scores\n"
                 f"for {dataset_labels.get(base_dataset_name, base_dataset_name)} {title_suffix}")
    ax.set_title(title, fontsize=14)

    if len(datasets) > 1:
        ax.legend(loc="upper right" if stats_loc == "upper left" else "upper left",
                  frameon=True, fancybox=True, framealpha=0.9, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)

    if all_data:
        stats_lines = [f"{lbl}: n={len(d):,}" if len(datasets) > 1 else f"n = {len(d):,}"
                       for d, lbl in zip(all_data, all_labels)]
        at = AnchoredText("\n".join(stats_lines), loc=stats_loc, frameon=True,
                          prop=dict(size=9),
                          bbox_to_anchor=(0.01, 0.0, 1.0, 1.0),
                          bbox_transform=ax.transAxes)
        at.patch.set_boxstyle("round,pad=0.3")
        at.patch.set_facecolor("white")
        at.patch.set_alpha(0.8)
        ax.add_artist(at)

    plt.tight_layout()

    save_dir = (output_dir or os.path.join(data_base_dir, base_dataset_name, "charts"))
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{save_name}_loss_scores.png")
    _save(out)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Edgotype bootstrap (uncontrolled)
# ---------------------------------------------------------------------------

def compute_edgotype_bootstrap(base_dataset_name, datasets, data_base_dir,
                                n_partners_suffix="", n_bootstrap=100_000,
                                random_seed=42,
                                posterior_threshold=edgotypes.DEFAULT_THRESHOLD):
    edgotype_classes = list(edgotypes.EDGOTYPES)
    np.random.seed(random_seed)

    mean_n_partners_path = (
        f"{data_base_dir}/{base_dataset_name}/mean_n_partners{n_partners_suffix}.npy"
    )
    if os.path.exists(mean_n_partners_path):
        mean_n_partners = np.load(mean_n_partners_path)
    else:
        mean_n_partners = None

    all_densities, all_bootstrap_densities, all_sample_ns = [], [], []

    for dataset in datasets:
        group = edgotypes.load_group(data_base_dir, base_dataset_name, dataset)
        if group is None:
            print(f"  Warning: missing "
                  f"{edgotypes.group_path(data_base_dir, base_dataset_name, dataset)}")
            continue

        data = group.classify(posterior_threshold)
        densities, counts = [], []
        total = len(data)
        for ec in edgotype_classes:
            c = np.sum(data == ec)
            counts.append(c)
            densities.append(c / total if total else 0)
        all_densities.append(densities)
        all_sample_ns.append(counts)

        # Use multinomial sampling (equivalent to bootstrap but O(n_bootstrap) not O(n*n_bootstrap))
        probs = np.array(counts, dtype=float) / total if total else np.ones(len(edgotype_classes)) / len(edgotype_classes)
        boot = np.random.multinomial(total, probs, size=n_bootstrap) / total
        all_bootstrap_densities.append(boot)

    return all_densities, all_bootstrap_densities, all_sample_ns


def run_comprehensive_analysis(data_base_dir, n_bootstrap=100_000, random_seed=42):
    print("=" * 70)
    print("COMPREHENSIVE EDGOTYPE ANALYSIS (uncontrolled bootstrap)")
    print("=" * 70)
    all_bootstrap = {}
    all_sample_ns = {}

    for cfg in CONFIGURATIONS:
        base, datasets, _, save_name, _, n_partners_suffix = cfg
        print(f"  {save_name} ...")
        _, boot, ns = compute_edgotype_bootstrap(
            base, datasets, data_base_dir,
            n_partners_suffix=n_partners_suffix,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed,
        )
        all_bootstrap[save_name] = boot
        all_sample_ns[save_name] = ns

    return all_bootstrap, all_sample_ns


# ---------------------------------------------------------------------------
# Enrichment plot
# ---------------------------------------------------------------------------

def calc_enrichment(f_obs, f_gnomad):
    return (f_obs - f_gnomad) / (f_obs + f_gnomad)


def _get_enrich_color(dataset, label):
    if dataset == "cosmic_onco":
        return {"Single": "#FFE0B2", "≥2": "#FFAB40", "≥4": "#FF6D00",
                "≥8": "#FF3D00", "≥16": "#BF360C", "≥32": "#7E2010"}.get(label, "#666")
    if dataset == "cosmic_tsg":
        return {"Single": "#FFFDE7", "≥2": "#FFD54F", "≥4": "#FFA000",
                "≥8": "#FF6E40", "≥16": "#EF5350", "≥32": "#C62828"}.get(label, "#666")
    color_map = {
        "HGMD": "#E74C3C",
        "Benign": "#1976D2", "Rare Benign": "#0D47A1", "Pathogenic": "#D32F2F", "VUS": "#9E9E9E",
        "Single": "#FFF59D", "≥2": "#FFCA28", "≥4": "#FF8F00",
        "≥8": "#FF5722", "≥16": "#E64A19", "≥32": "#B71C1C",
        "AF ≤ 1e-6": "#FF7043", "1e-6 < AF ≤ 1e-5": "#FFA726",
        "1e-5 < AF ≤ 1e-4": "#FFCA28", "1e-4 < AF ≤ 1e-3": "#9CCC65",
        "1e-3 < AF ≤ 1e-2": "#66BB6A", "1e-2 < AF": "#4CAF50",
        "ASD Case": "#FF9800", "NDD Case": "#9C27B0", "NDD Control": "#607D8B",
        "AR": "#00897B", "AD": "#FFB300",
        "Pathogenic AR": "#00897B", "Pathogenic AD": "#FFB300",
    }
    return color_map.get(label, "#666666")


def plot_enrichment_bootstrap(bootstrap_densities, sample_ns, output_dir,
                               suffix="sufficient_partners", controlled=0,
                               datasets=None, separate_components=False,
                               notsignif_alpha=0.9):
    gnomad_baseline = bootstrap_densities["gnomad"][0]
    n_bootstrap = gnomad_baseline.shape[0]

    enrichments_bootstrap = {}
    for dataset, subgroup_list in bootstrap_densities.items():
        if dataset in ("gnomad", "gnomad_disease_gene"):
            continue
        enrichments_bootstrap[dataset] = [
            np.array([calc_enrichment(sg[bi, :], gnomad_baseline[bi, :])
                      for bi in range(n_bootstrap)])
            for sg in subgroup_list
        ]

    default_order = ["clinvar", "cosmic", "cosmic_onco", "cosmic_tsg",
                     "hgmd", "gnomad_af", "neurodev", "asd"]
    dataset_order = datasets if datasets is not None else default_order

    n_tests = sum(len(v) for v in enrichments_bootstrap.values()) * 2
    print(f"  Bonferroni n_tests = {n_tests}")

    # Dynamic width
    span = sum(
        len([dataset_enrichment_labels[d]] if isinstance(dataset_enrichment_labels.get(d), str)
            else dataset_enrichment_labels.get(d, [])) + 1.5
        for d in dataset_order if d in enrichments_bootstrap
    )
    scaled_w = max(min(18 * span / 35.0, 18), 6)

    component_names = ["Quasi-Null", "Edgetic"]
    component_indices = [1, 2]

    dataset_display_names = {
        "clinvar": "ClinVar",
        "cosmic": "COSMIC",
        "cosmic_onco": "COSMIC (Onco)", "cosmic_tsg": "COSMIC (TSG)",
        "hgmd": "HGMD", "gnomad_af": "gnomAD",
        "neurodev": "NDD", "asd": "ASD",
    }

    # Build shared x-axis metadata
    shared_meta = {"x_ticks": [], "x_labels": [], "boundaries": [], "final_x": 0}
    x_pos = 0
    for dataset in dataset_order:
        if dataset not in enrichments_bootstrap or dataset not in dataset_enrichment_labels:
            continue
        dataset_start = x_pos
        labels = dataset_enrichment_labels[dataset]
        if isinstance(labels, str):
            labels = [labels]
        for i, lbl in enumerate(labels):
            pub_lbl = lbl.replace("Case", "case").replace("Control", "control")
            ns_key = dataset
            n_str = (f"n={sum(sample_ns[ns_key][i]):,}"
                     if ns_key in sample_ns and i < len(sample_ns[ns_key])
                     else "n=0")
            shared_meta["x_labels"].append(f"{pub_lbl} ({n_str})")
            shared_meta["x_ticks"].append(x_pos)
            x_pos += 1
        shared_meta["boundaries"].append((dataset_start, x_pos - 0.5,
                                          dataset_display_names.get(dataset, dataset)))
        x_pos += 1.5
    shared_meta["final_x"] = x_pos

    def draw_component(ax, comp_idx, plot_idx, single_plot=False):
        xp = 0
        for dataset in dataset_order:
            if dataset not in enrichments_bootstrap or dataset not in dataset_enrichment_labels:
                continue
            labels = dataset_enrichment_labels[dataset]
            if isinstance(labels, str):
                labels = [labels]
            for sg_idx, enrich_arr in enumerate(enrichments_bootstrap[dataset]):
                vals = enrich_arr[:, comp_idx]
                median = np.median(vals)
                p16, p84 = np.percentile(vals, [16, 84])
                alpha_bonf = 0.05 / n_tests
                if median >= 0:
                    sig_bonf = np.percentile(vals, 100 * alpha_bonf) > 0
                    sig_uncorr = np.percentile(vals, 5) > 0
                else:
                    sig_bonf = np.percentile(vals, 100 * (1 - alpha_bonf)) < 0
                    sig_uncorr = np.percentile(vals, 95) < 0
                color = _get_enrich_color(dataset, labels[sg_idx])
                ax.bar(xp, median, width=0.8, color=color, edgecolor="black",
                       linewidth=1.2, alpha=0.9 if sig_bonf else notsignif_alpha)
                ax.errorbar(xp, median, yerr=[[median - p16], [p84 - median]],
                            fmt="none", ecolor="black", capsize=4, linewidth=2, alpha=0.7)
                if sig_bonf:
                    y_mark = (p84 + 0.05) if median > 0 else (p16 - 0.05)
                    ax.text(xp, y_mark, "*", ha="center",
                            va="bottom" if median > 0 else "top",
                            fontsize=20, fontweight="bold", color="black")
                elif sig_uncorr:
                    y_mark = (p84 + 0.05) if median > 0 else (p16 - 0.05)
                    ax.text(xp, y_mark, "•", ha="center",
                            va="bottom" if median > 0 else "top",
                            fontsize=12, fontweight="bold", color="black")
                xp += 1
            xp += 1.5
        if datasets is None and not single_plot:
            ax.text(-0.039, 0.99, f"({chr(65 + plot_idx)})", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", va="top", ha="right")
        for start, end, name in shared_meta["boundaries"]:
            mid = (start + end) / 2
            y_text = 0.90 if (single_plot or plot_idx == 0) else 1.02
            ax.text(mid, y_text, name, ha="center", va="bottom",
                    fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9,
                              edgecolor="gray", linewidth=1.5),
                    transform=ax.get_xaxis_transform())
        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_ylabel(f"{component_names[plot_idx]} Enrichment", fontsize=16, fontweight="bold")
        ax.set_xticks(shared_meta["x_ticks"])
        ax.set_xticklabels(shared_meta["x_labels"], rotation=45, ha="right", fontsize=13)
        ax.set_ylim([-1, 1.05])
        ax.set_xlim(-1.5, shared_meta["final_x"] - 2 + 1.5)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.3, linewidth=0.8)

    k_str = f"_k{controlled}" if controlled else ""
    base_name = f"enrichment_bootstrap_{suffix}{k_str}"

    if not separate_components:
        fig, axes = plt.subplots(2, 1, figsize=(scaled_w, 10),
                                 gridspec_kw={"hspace": 0, "height_ratios": [1, 1]})
        for pi, (ci, ax) in enumerate(zip(component_indices, axes)):
            draw_component(ax, ci, pi, single_plot=False)
            if pi == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, top=False)
                ax.set_yticks([-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
                ax.spines["bottom"].set_visible(False)
            else:
                ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
                ax.spines["top"].set_visible(False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        out = os.path.join(output_dir, f"{base_name}.png")
        _save(out)
        plt.close()
        print(f"  Saved: {out}")
    else:
        for pi, ci in enumerate(component_indices):
            fig, ax = plt.subplots(1, 1, figsize=(scaled_w, 6))
            draw_component(ax, ci, pi, single_plot=True)
            ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            plt.tight_layout()
            comp_slug = component_names[pi].lower().replace("-", "_").replace(" ", "_")
            out = os.path.join(output_dir, f"{base_name}_{comp_slug}.png")
            _save(out)
            plt.close()
            print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Partner-count-controlled bootstrap
# ---------------------------------------------------------------------------

def _batch_bootstrap_controlled(valid_variants, k, posterior_threshold,
                                  edgotype_classes, n_iterations):
    # Pre-convert partner scores to binary disruption arrays once
    binary_arrays = [np.array(v) > posterior_threshold for v in valid_variants]
    n_variants = len(binary_arrays)
    qwt_idx = edgotype_classes.index("Quasi-wild-type")
    qn_idx  = edgotype_classes.index("Quasi-null")
    ed_idx  = edgotype_classes.index("Edgetic")

    results = []
    for _ in range(n_iterations):
        # Bootstrap resample variants
        vi = np.random.choice(n_variants, size=n_variants, replace=True)
        qwt = qn = ed = 0
        for idx in vi:
            b = binary_arrays[idx]
            n_avail = len(b)
            chosen = b[np.random.choice(n_avail, size=min(k, n_avail), replace=False)]
            nd = int(np.sum(chosen))
            if nd == k:
                qn += 1
            elif nd == 0:
                qwt += 1
            else:
                ed += 1
        row = [0.0] * len(edgotype_classes)
        row[qwt_idx] = qwt / n_variants
        row[qn_idx]  = qn  / n_variants
        row[ed_idx]  = ed  / n_variants
        results.append(row)
    return np.array(results)


def controlled_bootstrap_for_dataset(dataset, base, data_base_dir, n_partners_suffix,
                                      k=3, posterior_threshold=0.5,
                                      n_bootstrap=10_000, n_jobs=-1):
    edgotype_classes = list(edgotypes.EDGOTYPES)
    group = edgotypes.load_group(data_base_dir, base, dataset)
    if group is None:
        print(f"  Warning: missing {edgotypes.group_path(data_base_dir, base, dataset)}")
        return None, None

    posterior_data = group.scores_by_variant()
    valid = [v for v in posterior_data if len(v) >= k]
    if not valid:
        print(f"  Warning: no variants with >={k} partners in {dataset}")
        return [0, 0, 0], np.zeros((n_bootstrap, 3))

    # Cap for speed; set high enough that no current dataset is subsampled
    # (gnomAD has ~163K k3-valid variants; 200K avoids any capping)
    MAX_VARIANTS = 200_000
    if len(valid) > MAX_VARIANTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(valid), size=MAX_VARIANTS, replace=False)
        valid = [valid[i] for i in idx]
        print(f"    {dataset}: subsampled to {MAX_VARIANTS} variants for bootstrap speed")
    else:
        print(f"    {dataset}: {len(valid)}/{len(posterior_data)} variants with >={k} partners")

    # Observed edgotypes
    np.random.seed(42)
    obs = []
    for vp in valid:
        s = np.random.choice(vp, size=k, replace=False)
        frac = np.sum(s >= posterior_threshold) / k
        obs.append("Quasi-wild-type" if frac == 0 else ("Quasi-null" if frac == 1 else "Edgetic"))
    obs = np.array(obs)
    counts = [int(np.sum(obs == ec)) for ec in edgotype_classes]

    # Parallel bootstrap
    n_cores = n_jobs if n_jobs > 0 else os.cpu_count()
    batch_size = max(1, (n_bootstrap // n_cores) + 1)
    batches = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_batch_bootstrap_controlled)(valid, k, posterior_threshold,
                                              edgotype_classes, batch_size)
        for _ in range(n_cores)
    )
    boot = np.vstack(batches)[:n_bootstrap]
    return counts, boot


def run_controlled_bootstrap_analysis(data_base_dir, k=3, n_bootstrap=10_000, n_jobs=-1):
    print(f"  k={k} partner-controlled bootstrap …")
    all_bootstrap, all_sample_ns = {}, {}

    for cfg in CONFIGURATIONS:
        base, datasets, _, save_name, _, n_partners_suffix = cfg
        if save_name == "gnomad_disease_gene":
            continue
        print(f"    {save_name}")
        ns_list, boot_list = [], []
        for dataset in datasets:
            counts, boot = controlled_bootstrap_for_dataset(
                dataset, base, data_base_dir, n_partners_suffix,
                k=k, n_bootstrap=n_bootstrap, n_jobs=n_jobs
            )
            if counts is not None:
                ns_list.append(counts)
                boot_list.append(boot)
        if ns_list:
            all_bootstrap[save_name] = boot_list
            all_sample_ns[save_name] = ns_list

    return all_bootstrap, all_sample_ns


# ---------------------------------------------------------------------------
# Tumor site analysis
# ---------------------------------------------------------------------------

def _get_density(edgotypes):
    classes = list(edgotypes.EDGOTYPES)
    total = len(edgotypes)
    return [np.sum(np.array(edgotypes) == ec) / total if total else 0 for ec in classes]


def plot_tumor_site_edgotypes(tumor_site_to_edgotypes, cosmic_label, save_dir,
                               min_data_pts=20, colormap="viridis", max_sites=999):
    single_variants = cosmic_label.endswith("single")
    title_tag = "Single Occurrence" if single_variants else "Recurrence ≥ 4"

    sorted_sites = sorted(tumor_site_to_edgotypes, key=lambda k: len(tumor_site_to_edgotypes[k]),
                          reverse=True)
    data_filtered = [(site, _get_density(tumor_site_to_edgotypes[site]),
                      len(tumor_site_to_edgotypes[site]))
                     for site in sorted_sites
                     if len(tumor_site_to_edgotypes[site]) >= min_data_pts]
    data_filtered.sort(key=lambda x: x[1][0], reverse=True)
    if len(data_filtered) > max_sites:
        data_filtered = data_filtered[:max_sites]

    if not data_filtered:
        print(f"  No tumor sites with >={min_data_pts} samples — skipping {cosmic_label}")
        return

    categories = ["Quasi-Wild-Type", "Quasi-Null", "Edgetic"]
    x = np.arange(len(categories))
    n_groups = len(data_filtered)
    width = 0.8 / n_groups
    cmap = cm.get_cmap(colormap, n_groups)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (site, densities, size) in enumerate(data_filtered):
        offset = -0.4 + width * i + width / 2
        label = f"{site[:15].replace('_', ' ').strip()}{'–' if len(site) > 15 else ''} (n={size:,})"
        ax.bar(x + offset, densities, width, label=label,
               color=cmap(i), edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Distribution of Edgotype Classes Across Tumor Sites\n"
                 f"COSMIC {title_tag} Variants", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=True,
              framealpha=0.9, ncol=4)
    plt.tight_layout()
    out = os.path.join(save_dir, f"cosmic_{cosmic_label}_tumor_site_edgotypes.png")
    _save(out)
    plt.close()
    print(f"  Saved: {out}")


def _trend_statistic(tumor_site_to_edgotypes, min_data_pts):
    classes = list(edgotypes.EDGOTYPES)
    site_props = [
        [np.sum(np.array(v) == ec) / len(v) for ec in classes]
        for v in tumor_site_to_edgotypes.values() if len(v) >= min_data_pts
    ]
    if len(site_props) < 2:
        return 0.0
    return float(np.sum(np.std(np.array(site_props), axis=0)))


def run_tumor_site_analysis(data_base_dir, output_dir, min_data_pts=20, n_permutations=10_000):
    print("  Running tumor site variation analysis …")

    paths = {
        "4+": os.path.join(data_base_dir, "cosmic/cosmic_4+_tumor_site_to_edgotypes.pkl"),
        "single": os.path.join(data_base_dir, "cosmic/cosmic_single_tumor_site_to_edgotypes.pkl"),
    }
    data = {}
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping tumor site analysis")
            return
        with open(path, "rb") as f:
            data[label] = pickle.load(f)

    save_dir = os.path.join(output_dir, "cosmic_tumor_sites", "charts")
    os.makedirs(save_dir, exist_ok=True)

    # Bar charts
    plot_tumor_site_edgotypes(data["4+"], "4+", save_dir,
                               min_data_pts=min_data_pts, max_sites=999)
    plot_tumor_site_edgotypes(data["single"], "single", save_dir,
                               min_data_pts=min_data_pts, max_sites=999)

    # Permutation test
    obs_diff = _trend_statistic(data["4+"], min_data_pts) - _trend_statistic(data["single"], min_data_pts)
    all_sites = set(data["4+"].keys()) | set(data["single"].keys())
    perm_diffs = []
    for _ in range(n_permutations):
        perm_r, perm_s = {}, {}
        for site in all_sites:
            rec = list(data["4+"].get(site, []))
            sin = list(data["single"].get(site, []))
            combined = rec + sin
            np.random.shuffle(combined)
            perm_r[site] = combined[:len(rec)]
            perm_s[site] = combined[len(rec):]
        perm_diffs.append(_trend_statistic(perm_r, min_data_pts) -
                          _trend_statistic(perm_s, min_data_pts))
    perm_diffs = np.array(perm_diffs)
    p_value = np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) / n_permutations
    print(f"  Observed variation difference: {obs_diff:.4f}, p = {p_value:.4f}")

    # Permutation histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(perm_diffs, bins=50, alpha=0.85, edgecolor="#333333",
            color="#B0BEC5", linewidth=0.5)
    ax.axvline(obs_diff, color="#D32F2F", linestyle="--", linewidth=2.5,
               label=f"Observed: {obs_diff:.3f}")
    ax.axvline(0, color="#424242", linestyle="-", alpha=0.5, linewidth=1)
    ax.set_xlabel("Variation Difference (Recurrent − Single)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Permutation Test for Edgotype Variation\nAcross Tumor Sites",
                 fontsize=13, pad=15)
    p_str = "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"
    ax.text(0.03, 0.97, f"n = {n_permutations:,} permutations\n{p_str}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(save_dir, "permutation_histogram.png")
    _save(out)
    plt.close()
    print(f"  Saved: {out}")

    # CV comparison (recurrent vs single)
    classes = list(edgotypes.EDGOTYPES)
    rec_props = {ec: [] for ec in classes}
    sin_props = {ec: [] for ec in classes}
    for site in data["4+"]:
        v = data["4+"][site]
        if len(v) < min_data_pts:
            continue
        total = len(v)
        for ec in classes:
            rec_props[ec].append(np.sum(np.array(v) == ec) / total)
    for site in data["single"]:
        v = data["single"][site]
        if len(v) < min_data_pts:
            continue
        total = len(v)
        for ec in classes:
            sin_props[ec].append(np.sum(np.array(v) == ec) / total)

    def _cv(vals):
        vals = np.array(vals)
        return np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else 0

    rec_cvs = [_cv(rec_props[ec]) for ec in classes]
    sin_cvs = [_cv(sin_props[ec]) for ec in classes]
    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - w / 2, rec_cvs, w, label="Recurrence ≥ 4",
           color="#FF5722", edgecolor="#F4511E", linewidth=1.5, alpha=0.9)
    ax.bar(x + w / 2, sin_cvs, w, label="Single Occurrence",
           color="#FFF59D", edgecolor="#FDD835", linewidth=1.5, alpha=0.9)
    for bars, cvs in [((x - w / 2), rec_cvs), ((x + w / 2), sin_cvs)]:
        for xi, cv in zip(bars, cvs):
            ax.text(xi, cv + 0.01, f"{cv:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Edgotype Class", fontsize=12)
    ax.set_ylabel("Coefficient of Variation", fontsize=12)
    ax.set_title("Variation in Edgotype Proportions\nAcross Tumor Sites", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(save_dir, "cv_comparison.png")
    _save(out)
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global _DEMO_STAMP
    _DEMO_STAMP = bool(getattr(args, "demo_tier", False))
    if _DEMO_STAMP:
        print("*** DEMONSTRATION TIER: every figure will be stamped as NOT the "
              "published all-data numbers. ***", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.score_histograms:
        print("=== Score distribution histograms ===")
        for cfg in HISTOGRAM_CONFIGS:
            base, datasets, stats_loc, save_name, alpha = cfg
            plot_multi_class_histogram(
                base_dataset_name=base,
                datasets=datasets,
                data_base_dir=args.data_dir,
                save_name=save_name,
                stats_loc=stats_loc,
                alpha=alpha,
                output_dir=os.path.join(args.output_dir, base, "charts"),
            )

    if args.edgotype_bootstrap:
        print("=== Uncontrolled edgotype bootstrap ===")
        cache_path = os.path.join(args.output_dir, "all_bootstrap_results.pkl")
        if os.path.exists(cache_path) and not args.recompute:
            print(f"  Loading cached bootstrap from {cache_path}")
            with open(cache_path, "rb") as f:
                all_bootstrap = pickle.load(f)
            all_sample_ns, _ = _quick_sample_counts(args.data_dir)
        else:
            all_bootstrap, all_sample_ns = run_comprehensive_analysis(
                args.data_dir, n_bootstrap=args.n_bootstrap
            )
            with open(cache_path, "wb") as f:
                pickle.dump(all_bootstrap, f)
            print(f"  Saved bootstrap cache: {cache_path}")

        print("  Plotting enrichment …")
        plot_enrichment_bootstrap(
            all_bootstrap, all_sample_ns, args.output_dir,
            suffix="sufficient_partners", separate_components=args.separate_components,
        )
        plot_enrichment_bootstrap(
            all_bootstrap, all_sample_ns, args.output_dir,
            suffix="sufficient_partners_clinvar_hgmd",
            datasets=["clinvar", "hgmd"],
            separate_components=False,
        )

    if args.controlled_bootstrap:
        print("=== Partner-controlled bootstrap ===")
        k_values = [3] if args.k3_only else [3, 5, 7]
        for k in k_values:
            cache_path = os.path.join(args.output_dir, f"bootstrap_results_controlled_k{k}.pkl")
            ns_cache_path = os.path.join(args.output_dir, f"sample_ns_controlled_k{k}.pkl")
            if os.path.exists(cache_path) and os.path.exists(ns_cache_path) and not args.recompute:
                print(f"  Loading k={k} cache …")
                with open(cache_path, "rb") as f:
                    boot_k = pickle.load(f)
                with open(ns_cache_path, "rb") as f:
                    ns_k = pickle.load(f)
            else:
                boot_k, ns_k = run_controlled_bootstrap_analysis(
                    args.data_dir, k=k, n_bootstrap=args.n_bootstrap, n_jobs=args.n_jobs
                )
                with open(cache_path, "wb") as f:
                    pickle.dump(boot_k, f)
                with open(ns_cache_path, "wb") as f:
                    pickle.dump(ns_k, f)
                print(f"  Saved k={k} cache.")

            print(f"  Plotting k={k} enrichment …")
            plot_enrichment_bootstrap(
                boot_k, ns_k, args.output_dir,
                suffix="sufficient_partners", controlled=k,
            )

    if args.tumor_sites:
        print("=== Tumor site analysis ===")
        run_tumor_site_analysis(
            args.data_dir, args.output_dir,
            min_data_pts=args.min_tumor_site_pts,
            n_permutations=args.n_permutations,
        )

    print("\nDone.")


def _quick_sample_counts(data_base_dir):
    edgotype_classes = list(edgotypes.EDGOTYPES)
    all_ns, all_dens = {}, {}
    for cfg in CONFIGURATIONS:
        base, datasets, _, save_name, _, _ = cfg
        ns_list, dens_list = [], []
        for dataset in datasets:
            group = edgotypes.load_group(data_base_dir, base, dataset)
            if group is None:
                continue
            data = group.classify()
            total = len(data)
            counts = [int(np.sum(data == ec)) for ec in edgotype_classes]
            dens = [c / total if total else 0 for c in counts]
            ns_list.append(counts)
            dens_list.append(dens)
        all_ns[save_name] = ns_list
        all_dens[save_name] = dens_list
    return all_ns, all_dens


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate publication charts for variant database analysis"
    )
    p.add_argument("--demo-tier", action="store_true",
                   help="stamp every figure as coming from the Sahni+Fragoza "
                        "demonstration model rather than the published all-data "
                        "model (see run_variant_db_inference.py --model-tier)")
    p.add_argument("--data-dir", required=True,
                   help="Base results directory (e.g., variant_dbs_sufficient_partners)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for figures and cache PKLs (default: --data-dir)")
    p.add_argument("--n-bootstrap", type=int, default=100_000,
                   help="Bootstrap iterations (default: 100000)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for controlled bootstrap (-1 = all CPUs)")
    p.add_argument("--score-histograms", action="store_true",
                   help="Generate score distribution histograms")
    p.add_argument("--edgotype-bootstrap", action="store_true",
                   help="Run uncontrolled edgotype enrichment bootstrap")
    p.add_argument("--controlled-bootstrap", action="store_true",
                   help="Run partner-count-controlled edgotype bootstrap (k=3,5,7)")
    p.add_argument("--k3-only", action="store_true",
                   help="Only run k=3 controlled bootstrap (faster)")
    p.add_argument("--tumor-sites", action="store_true",
                   help="Run COSMIC tumor-site variation analysis")
    p.add_argument("--separate-components", action="store_true",
                   help="Save Quasi-Null and Edgetic enrichment as separate figures")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute bootstrap even if cache PKL exists")
    p.add_argument("--min-tumor-site-pts", type=int, default=20,
                   help="Minimum variants per tumor site (default: 20)")
    p.add_argument("--n-permutations", type=int, default=10_000,
                   help="Permutations for tumor site variation test (default: 10000)")

    parsed = p.parse_args()
    if parsed.output_dir is None:
        parsed.output_dir = parsed.data_dir
    main(parsed)
