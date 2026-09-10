#!/usr/bin/env python
"""Classify variant database predictions into edgotype classes.

Reads raw MutPred-PPI prediction TSVs (complex_id, variant, score) and variant
subset metadata to produce _edgotype_classes.npy and _posterior_ls.pkl files
needed by variant_db_charts.py.

Edgotype classification (per unique variant, across all tested partners):
  Quasi-null      : all partner scores > threshold (all disrupted)
  Quasi-wild-type : all partner scores <= threshold (all preserved)
  Edgetic         : mixed

Output directory structure:
  {output_dir}/clinvar/    benign, pathogenic, vus, rare_benign, ar_pathogenic, ad_pathogenic
  {output_dir}/gnomad/     gnomad, gnomad_upper_af_1e-06, ..., gnomad_upper_af_0.1
  {output_dir}/hgmd/       hgmd, ar_hgmd, ad_hgmd
  {output_dir}/cosmic/     cosmic_single, cosmic_2+, ..., cosmic_32+,
                           cosmic_onco_*, cosmic_tsg_*
  {output_dir}/fu_autism/  fu_autism

ar_pathogenic/ad_pathogenic and ar_hgmd/ad_hgmd stratify Pathogenic/HGMD variants by
whether their gene has an autosomal-recessive-only or autosomal-dominant-only mode of
inheritance (ClinGen Gene-Disease Validity curations; see build_ar_ad_gene_sets.py).
These require $MUTPRED_DATA_ROOT/clingen_ar_ad_uniprot_sets.pkl.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT, VARIANT_DBS_DIR  # noqa: E402


# ── paths ──────────────────────────────────────────────────────────────────────

_BASE = str(DATA_ROOT)
_HOME = f"{_BASE}/home"

# Predictions from the SFVCFP model (Sahni+Fragoza+VarChAMP), which is what the
# manuscript's variant-repository figures report. Overridable with --pred-dir,
# but every database in a run must come from one model: mixing them silently
# produced a Fig 5 in which ClinVar/HGMD were scored by the SF model while
# gnomAD/COSMIC/autism were scored by SFVCFP.
DEFAULT_PRED_DIR = VARIANT_DBS_DIR

SUBSET_FILES = {
    "clinvar": {
        "pathogenic": str(ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl"),
        "benign":     str(ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl"),
        "vus":        str(ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl"),
    },
    "hgmd": {
        "hgmd":       str(ANNOTATIONS_LICENSED_DIR / "hgmd_variant_subset.pkl"),
    },
    "fu_autism": {
        "fu_autism":  str(ANNOTATIONS_DIR / "autism" / "variant_subset.pkl"),
    },
}

NEURODEV_LABEL_FILE = str(ANNOTATIONS_DIR / "autism" / "variant_label_dict.pkl")  # {uniprot} {variant} -> 0 (control) or 1 (case)

GNOMAD_AF_FILE = str(ANNOTATIONS_DIR / "gnomad_allele_frequencies.tsv")
GNOMAD_AF_THRESHOLDS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]  # upper bounds of exclusive bins

BENIGN_AF_FILE = str(ANNOTATIONS_DIR / "benign_allele_frequencies.tsv")
RARE_BENIGN_AF_THRESHOLD = 0.01

COSMIC_TUMOR_SITE_FILE = str(ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl")  # recurrence = len(sites)
COSMIC_ONCO_TSG_FILE   = str(ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl")
COSMIC_RECURRENCE_BINS = [1, 2, 4, 8, 16, 32]  # "single" = 1; "2+" = >=2, etc.

AR_AD_UNIPROT_FILE = str(ANNOTATIONS_DIR / "clingen_ar_ad_uniprot_sets.pkl")  # {"AR": set[uniprot], "AD": set[uniprot]}


# ── core helpers ───────────────────────────────────────────────────────────────

def load_predictions(tsv_path):
    """Load TSV into dict: (uniprot, variant, partner) -> score.

    Reads the current explicit-column schema
    (`interactor  partner  mutation  score`) and the legacy composite one
    (`complex_id  variant  score`, where `complex_id` is `{uniprot}_{partner}`).
    Only the reader is schema-aware; nothing downstream of this function changed.
    """
    pairs = {}
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        legacy = header[:1] == ["complex_id"]
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if legacy:
                if len(parts) < 3:
                    continue
                complex_id, variant, score = parts[0], parts[1], float(parts[2])
                # UniProt accessions contain no underscore, so the first one is
                # the delimiter.
                under = complex_id.index("_")
                uniprot, partner = complex_id[:under], complex_id[under + 1:]
            else:
                if len(parts) < 4:
                    continue
                uniprot, partner, variant, score = (
                    parts[0], parts[1], parts[2], float(parts[3]))
            pairs[(uniprot, variant, partner)] = score
    return pairs


def group_by_variant(pairs):
    """Group (uniprot, variant, partner)->score into (uniprot, variant) -> {partner: score}."""
    grouped = defaultdict(dict)
    for (uniprot, variant, partner), score in pairs.items():
        grouped[(uniprot, variant)][partner] = score
    return grouped


def classify_edgotype(scores, threshold=0.5):
    """Return 'Quasi-null', 'Quasi-wild-type', or 'Edgetic' given a list of scores."""
    scores = list(scores)
    n_disrupted = sum(s > threshold for s in scores)
    if n_disrupted == len(scores):
        return "Quasi-null"
    elif n_disrupted == 0:
        return "Quasi-wild-type"
    else:
        return "Edgetic"


def load_ar_ad_uniprots():
    """Load mutually-exclusive AR-only/AD-only UniProt sets (see build_ar_ad_gene_sets.py)."""
    if not os.path.exists(AR_AD_UNIPROT_FILE):
        return None, None
    with open(AR_AD_UNIPROT_FILE, "rb") as f:
        d = pickle.load(f)
    return d.get("AR", set()), d.get("AD", set())


def build_arrays(grouped, subset, threshold=0.5, min_partners=1):
    """Build edgotype_classes array and posterior_ls list for a given variant subset.

    subset : set of (uniprot, variant, partner) tuples that belong to this group.
    Returns (edgotype_classes_array, posterior_ls).
    """
    # Index subset by (uniprot, variant) for fast lookup
    subset_by_vt = defaultdict(set)
    for (u, v, p) in subset:
        subset_by_vt[(u, v)].add(p)

    edgotype_classes = []
    posterior_ls = []

    for (uniprot, variant), partner_scores in grouped.items():
        # Filter to only partners in the subset
        allowed_partners = subset_by_vt.get((uniprot, variant), set())
        filtered_scores = {p: s for p, s in partner_scores.items() if p in allowed_partners}
        if not filtered_scores:
            continue

        scores_list = list(filtered_scores.values())
        edgotype_classes.append(classify_edgotype(scores_list, threshold))
        if len(scores_list) >= min_partners:
            posterior_ls.append(scores_list)

    return np.array(edgotype_classes, dtype=str), posterior_ls


def save_outputs(out_dir, name, edgotype_classes, posterior_ls):
    os.makedirs(out_dir, exist_ok=True)
    ec_path = os.path.join(out_dir, f"{name}_edgotype_classes.npy")
    pl_path = os.path.join(out_dir, f"{name}_posterior_ls.pkl")
    np.save(ec_path, edgotype_classes)
    with open(pl_path, "wb") as f:
        pickle.dump(posterior_ls, f)
    counts = {v: int(np.sum(edgotype_classes == v))
              for v in ["Quasi-null", "Edgetic", "Quasi-wild-type"]}
    print(f"  {name}: n={len(edgotype_classes)} | {counts} | posterior_ls n={len(posterior_ls)}")


# ── per-database processing ────────────────────────────────────────────────────

def process_clinvar(tsv_path, out_dir, threshold, min_partners):
    print("Processing ClinVar...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)
    for name, pkl_path in SUBSET_FILES["clinvar"].items():
        with open(pkl_path, "rb") as f:
            subset = pickle.load(f)
        ec, pl = build_arrays(grouped, subset, threshold, min_partners)
        save_outputs(out_dir, name, ec, pl)

    # Rare benign: ClinVar benign variants with gnomAD AF <= RARE_BENIGN_AF_THRESHOLD
    benign_af_dict = {}
    if os.path.exists(BENIGN_AF_FILE):
        with open(BENIGN_AF_FILE) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    benign_af_dict[parts[0]] = float(parts[1])
    with open(SUBSET_FILES["clinvar"]["benign"], "rb") as f:
        benign_subset = pickle.load(f)
    rare_benign_subset = set()
    for (u, v, p) in benign_subset:
        key = f"{u} {v}"
        if benign_af_dict.get(key, 1.0) <= RARE_BENIGN_AF_THRESHOLD:
            rare_benign_subset.add((u, v, p))
    if rare_benign_subset:
        ec, pl = build_arrays(grouped, rare_benign_subset, threshold, min_partners)
        save_outputs(out_dir, "rare_benign", ec, pl)
    else:
        print("  rare_benign: no variants found (check BENIGN_AF_FILE path)")

    # AR-only / AD-only disease-gene stratification of Pathogenic variants
    ar_uniprots, ad_uniprots = load_ar_ad_uniprots()
    if ar_uniprots is None:
        print(f"  ar_pathogenic/ad_pathogenic: skipped, {AR_AD_UNIPROT_FILE} not found "
              "(run build_ar_ad_gene_sets.py first)")
        return
    with open(SUBSET_FILES["clinvar"]["pathogenic"], "rb") as f:
        pathogenic_subset = pickle.load(f)
    ar_pathogenic = {(u, v, p) for (u, v, p) in pathogenic_subset if u in ar_uniprots}
    ad_pathogenic = {(u, v, p) for (u, v, p) in pathogenic_subset if u in ad_uniprots}
    ec, pl = build_arrays(grouped, ar_pathogenic, threshold, min_partners)
    save_outputs(out_dir, "ar_pathogenic", ec, pl)
    ec, pl = build_arrays(grouped, ad_pathogenic, threshold, min_partners)
    save_outputs(out_dir, "ad_pathogenic", ec, pl)


def process_hgmd(tsv_path, out_dir, threshold, min_partners):
    print("Processing HGMD...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)
    with open(SUBSET_FILES["hgmd"]["hgmd"], "rb") as f:
        subset = pickle.load(f)
    ec, pl = build_arrays(grouped, subset, threshold, min_partners)
    save_outputs(out_dir, "hgmd", ec, pl)

    # AR-only / AD-only disease-gene stratification of HGMD variants
    ar_uniprots, ad_uniprots = load_ar_ad_uniprots()
    if ar_uniprots is None:
        print(f"  ar_hgmd/ad_hgmd: skipped, {AR_AD_UNIPROT_FILE} not found "
              "(run build_ar_ad_gene_sets.py first)")
        return
    ar_hgmd = {(u, v, p) for (u, v, p) in subset if u in ar_uniprots}
    ad_hgmd = {(u, v, p) for (u, v, p) in subset if u in ad_uniprots}
    ec, pl = build_arrays(grouped, ar_hgmd, threshold, min_partners)
    save_outputs(out_dir, "ar_hgmd", ec, pl)
    ec, pl = build_arrays(grouped, ad_hgmd, threshold, min_partners)
    save_outputs(out_dir, "ad_hgmd", ec, pl)


def process_autism(tsv_path, out_dir, threshold, min_partners):
    print("Processing autism (fu_autism + neurodev)...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)

    # Fu et al. ASD cases
    with open(SUBSET_FILES["fu_autism"]["fu_autism"], "rb") as f:
        fu_subset = pickle.load(f)
    ec, pl = build_arrays(grouped, fu_subset, threshold, min_partners)
    save_outputs(out_dir, "fu_autism", ec, pl)

    # Neurodev NDD case/control (from variant_label_dict: 0=control, 1=case)
    if os.path.exists(NEURODEV_LABEL_FILE):
        with open(NEURODEV_LABEL_FILE, "rb") as f:
            label_dict = pickle.load(f)
        ndd_case_subset = set()
        ndd_control_subset = set()
        for (u, v, p) in pairs.keys():
            key = f"{u} {v}"
            label = label_dict.get(key)
            if label == 1:
                ndd_case_subset.add((u, v, p))
            elif label == 0:
                ndd_control_subset.add((u, v, p))
        neurodev_out = os.path.join(os.path.dirname(out_dir), "neurodev")
        if ndd_case_subset:
            ec, pl = build_arrays(grouped, ndd_case_subset, threshold, min_partners)
            save_outputs(neurodev_out, "ndd_case", ec, pl)
        if ndd_control_subset:
            ec, pl = build_arrays(grouped, ndd_control_subset, threshold, min_partners)
            save_outputs(neurodev_out, "ndd_control", ec, pl)
    else:
        print(f"  neurodev: label file not found at {NEURODEV_LABEL_FILE}")


def process_gnomad(tsv_path, out_dir, threshold, min_partners):
    print("Processing gnomAD...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)

    # Load AF data
    af_map = {}
    with open(GNOMAD_AF_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                af_map[parts[0]] = float(parts[1])

    # Build subset from all available pairs (no subset filter needed — all are gnomAD)
    all_gnomad_pairs = set(pairs.keys())

    # Overall gnomAD
    ec, pl = build_arrays(grouped, all_gnomad_pairs, threshold, min_partners)
    save_outputs(out_dir, "gnomad", ec, pl)

    # AF-stratified bins: exclusive ranges (lo < AF <= hi)
    # gnomad_upper_af_X contains only variants with prev_thresh < AF <= X
    sorted_thresholds = sorted(GNOMAD_AF_THRESHOLDS)
    prev_thresh = 0.0
    for af_thresh in sorted_thresholds:
        name = f"gnomad_upper_af_{af_thresh}"
        af_subset = set()
        for (u, v, p) in all_gnomad_pairs:
            key = f"{u} {v}"
            af = af_map.get(key, 1.0)
            if prev_thresh < af <= af_thresh:
                af_subset.add((u, v, p))
        if not af_subset:
            print(f"  {name}: no variants found, skipping")
            prev_thresh = af_thresh
            continue
        ec, pl = build_arrays(grouped, af_subset, threshold, min_partners)
        save_outputs(out_dir, name, ec, pl)
        prev_thresh = af_thresh


def process_cosmic(tsv_path, out_dir, threshold, min_partners):
    print("Processing COSMIC...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)
    all_cosmic_pairs = set(pairs.keys())

    # vt_to_tumor_site: "{uniprot} {variant}" -> list of tumor sites; recurrence = len(list)
    with open(COSMIC_TUMOR_SITE_FILE, "rb") as f:
        tumor_site_dict = pickle.load(f)
    recurrence_dict = {k: len(v) for k, v in tumor_site_dict.items()}

    # Onco/TSG sets: {role: set of "{uniprot} {variant}"} strings
    with open(COSMIC_ONCO_TSG_FILE, "rb") as f:
        onco_tsg = pickle.load(f)
    onco_vts = onco_tsg.get("oncogene", set())
    tsg_vts  = onco_tsg.get("TSG", set())

    def recurrence_subset(pairs_set, min_rec, max_rec=None, onco_tsg_filter=None):
        subset = set()
        for (u, v, p) in pairs_set:
            key = f"{u} {v}"
            rec = recurrence_dict.get(key, 0)
            if rec < min_rec:
                continue
            if max_rec is not None and rec > max_rec:
                continue
            if onco_tsg_filter == "oncogene" and key not in onco_vts:
                continue
            if onco_tsg_filter == "TSG" and key not in tsg_vts:
                continue
            subset.add((u, v, p))
        return subset

    # Overall recurrence bins
    bin_defs = [
        ("cosmic_single", 1, 1),
        ("cosmic_2+",  2, None),
        ("cosmic_4+",  4, None),
        ("cosmic_8+",  8, None),
        ("cosmic_16+", 16, None),
        ("cosmic_32+", 32, None),
    ]
    for name, min_rec, max_rec in bin_defs:
        subset = recurrence_subset(all_cosmic_pairs, min_rec, max_rec)
        if not subset:
            print(f"  {name}: no variants, skipping")
            continue
        ec, pl = build_arrays(grouped, subset, threshold, min_partners)
        save_outputs(out_dir, name, ec, pl)

    # Oncogene subsets
    for name, min_rec, max_rec in bin_defs:
        onco_name = name.replace("cosmic_", "cosmic_onco_")
        subset = recurrence_subset(all_cosmic_pairs, min_rec, max_rec, "oncogene")
        if not subset:
            print(f"  {onco_name}: no variants, skipping")
            continue
        ec, pl = build_arrays(grouped, subset, threshold, min_partners)
        save_outputs(out_dir, onco_name, ec, pl)

    # TSG subsets
    for name, min_rec, max_rec in bin_defs:
        tsg_name = name.replace("cosmic_", "cosmic_tsg_")
        subset = recurrence_subset(all_cosmic_pairs, min_rec, max_rec, "TSG")
        if not subset:
            print(f"  {tsg_name}: no variants, skipping")
            continue
        ec, pl = build_arrays(grouped, subset, threshold, min_partners)
        save_outputs(out_dir, tsg_name, ec, pl)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Classify variant DB predictions into edgotype classes")
    p.add_argument("--output-dir", required=True,
                   help="Base output directory (will create DB subdirs inside)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Disruption score threshold (default: 0.5)")
    p.add_argument("--min-partners", type=int, default=1,
                   help="Minimum partners for posterior_ls (default: 1)")
    p.add_argument("--databases", nargs="+",
                   choices=["clinvar", "gnomad", "hgmd", "cosmic", "fu_autism"],
                   default=["clinvar", "gnomad", "hgmd", "cosmic", "fu_autism"],
                   help="Databases to process (default: all)")
    p.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR),
                   help=f"Directory holding the per-database prediction TSVs "
                        f"(default: {DEFAULT_PRED_DIR}). All databases in one run "
                        f"must come from the same model.")
    args = p.parse_args()

    _DB_TSV_NAMES = {
        "clinvar":   "clinvar_mutpred_ppi_predictions.tsv",
        "gnomad":    "gnomad_mutpred_ppi_predictions.tsv",
        "hgmd":      "hgmd_mutpred_ppi_predictions.tsv",
        "cosmic":    "cosmic_mutpred_ppi_predictions.tsv",
        "fu_autism": "autism_mutpred_ppi_predictions.tsv",
    }
    db_funcs = {
        "clinvar":   (process_clinvar,  "clinvar"),
        "gnomad":    (process_gnomad,   "gnomad"),
        "hgmd":      (process_hgmd,     "hgmd"),
        "cosmic":    (process_cosmic,   "cosmic"),
        "fu_autism": (process_autism,   "fu_autism"),
    }
    print(f"Prediction TSVs: {args.pred_dir}", flush=True)

    for db in args.databases:
        func, out_subdir = db_funcs[db]
        tsv_path = os.path.join(args.pred_dir, _DB_TSV_NAMES[db])
        out_dir = os.path.join(args.output_dir, out_subdir)
        if not os.path.exists(tsv_path):
            print(f"WARNING: TSV not found: {tsv_path}, skipping {db}")
            continue
        func(tsv_path, out_dir, args.threshold, args.min_partners)

    print("\nDone.")


if __name__ == "__main__":
    main()
