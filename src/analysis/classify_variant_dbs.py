#!/usr/bin/env python
"""Classify variant database predictions into edgotype classes.

Reads MutPred-PPI prediction TSVs and variant-subset metadata, and writes one
tidy table per variant group -- `{output_dir}/{db}/{group}.csv.gz`, one row per
scored variant-partner pair. Edgotypes are derived from it on read via
`analysis.edgotypes`, so no stored artifact carries a baked-in threshold.

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
  {output_dir}/neurodev/  ndd_case, ndd_control
  {output_dir}/asd/       asd

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
import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
from paths import (ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT,  # noqa: E402
                   DATASETS_DIR, VARIANT_DBS_DIR)
VARIANT_ROWS_DIR = DATASETS_DIR / "variant_dbs"
from analysis import edgotypes  # noqa: E402


# ── paths ──────────────────────────────────────────────────────────────────────

_BASE = str(DATA_ROOT)
_HOME = f"{_BASE}/home"

# Predictions from the all-data model (Sahni+Fragoza+VarChAMP), which is what the
# manuscript's variant-repository figures report. Overridable with --pred-dir,
# but every database in a run must come from one model: mixing them silently
# produced a Fig 5 in which ClinVar/HGMD were scored by the SF model while
# gnomAD/COSMIC/neurodev were scored by an earlier partial model.
# run_variant_db_inference.py writes {DATA_ROOT}/{db}/mutpred_ppi_predictions.tsv,
# so that is what this reads. There is no collection step between the two.
DEFAULT_PRED_DIR = DATA_ROOT


def prediction_tsv(pred_dir, db):
    """Path to one database's prediction TSV.

    Two layouts are in use and both are legitimate: inference writes
    `{db}/mutpred_ppi_predictions.tsv` by default, and the reproduction notebook
    redirects every database into one directory as
    `{db}_mutpred_ppi_predictions.tsv` so a QUICK run stays out of `results/`.
    Prefer the per-database form, fall back to the collected one.
    """
    per_db = os.path.join(str(pred_dir), db, "mutpred_ppi_predictions.tsv")
    if os.path.exists(per_db):
        return per_db
    return os.path.join(str(pred_dir), f"{db}_mutpred_ppi_predictions.tsv")


def load_biogrid_partner_counts(db):
    """(uniprot, variant) -> number of partners BioGRID lists for it.

    Read from the canonical `{db}_rows.csv.gz`, which is the full enumeration of
    every variant-partner pair considered for the database -- including pairs
    that could not be scored. The per-group subset files cannot stand in for it:
    they carry group membership, and only some of them happen to enumerate every
    BioGRID partner, so taking the count from them would apply the manuscript's
    coverage rule inconsistently across databases.
    """
    rows = pd.read_csv(VARIANT_ROWS_DIR / f"{db}_rows.csv.gz",
                       usecols=["interactor", "partner", "mutation"])
    counts = rows.groupby(["interactor", "mutation"])["partner"].nunique()
    return {(u, v): int(n) for (u, v), n in counts.items()}

SUBSET_FILES = {
    "clinvar": {
        "pathogenic": str(ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl"),
        "benign":     str(ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl"),
        "vus":        str(ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl"),
    },
    "hgmd": {
        "hgmd":       str(ANNOTATIONS_LICENSED_DIR / "hgmd_variant_subset.pkl"),
    },
}

NEURODEV_LABEL_FILE = str(ANNOTATIONS_DIR / "neurodev" / "variant_label_dict.pkl")  # {uniprot} {variant} -> 0 (control) or 1 (case)

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


def load_ar_ad_uniprots():
    """Load mutually-exclusive AR-only/AD-only UniProt sets (see build_ar_ad_gene_sets.py)."""
    if not os.path.exists(AR_AD_UNIPROT_FILE):
        return None, None
    with open(AR_AD_UNIPROT_FILE, "rb") as f:
        d = pickle.load(f)
    return d.get("AR", set()), d.get("AD", set())


def build_arrays(grouped, subset, threshold=0.5, min_partners=1,
                 biogrid_counts=None):
    """Tidy table of scored variant-partner pairs for one variant group.

    subset : set of (uniprot, variant, partner) tuples belonging to this group.
             Only the (uniprot, variant) part is used -- membership of a clinical
             or cohort group is a property of the VARIANT, not of a pair.
    Returns a DataFrame with `analysis.edgotypes.COLUMNS`.

    The partner universe is the BioGRID direct-binding interactome, enforced once
    upstream: every pair in `{db}_rows.csv.gz` -- and so every scored pair -- is a
    direct-binding edge (verified: zero non-dirbind pairs in any database). Using
    the subset's own partner lists as a second filter here was wrong, because
    those files enumerate partners only incidentally. For ClinVar they happened to
    be complete and filtered nothing; for the ASD cohort they were partial and
    silently discarded 6,327 scored pairs -- every one of them a direct-binding
    edge -- leaving 1,111 of 7,438.

    `grouped` comes from the predictions TSV, so partners whose pair has no
    contact graph are already absent and each variant is described by the
    partners that could actually be scored.

    `min_partners` implements the coverage rule stated in the manuscript: a
    variant is analysed only when at least `min_partners` of its partners were
    tested, *and only when BioGRID lists that many for it in the first place*.
    A variant BioGRID knows one partner for is processed normally; one BioGRID
    lists twenty partners for, but which could only be scored on two, is dropped
    as too poorly covered to edgotype.

    `threshold` is not applied here. Edgotypes are derived on read, so the same
    table serves the default analysis and the threshold sweep without either
    going stale against the other.
    """
    group_variants = {(u, v) for (u, v, _partner) in subset}

    records = []
    for (uniprot, variant), partner_scores in grouped.items():
        if (uniprot, variant) not in group_variants:
            continue
        n_biogrid = (biogrid_counts or {}).get((uniprot, variant), len(partner_scores))
        if n_biogrid >= min_partners and len(partner_scores) < min_partners:
            continue
        for partner, score in partner_scores.items():
            records.append((uniprot, variant, partner, score, n_biogrid))

    return pd.DataFrame(records, columns=edgotypes.COLUMNS)


def save_outputs(out_dir, name, table):
    path = edgotypes.save_group(out_dir, name, table)
    group = edgotypes.EdgotypeGroup(name=name, table=table)
    print(f"  {name}: n={len(group)} variants, {len(table)} pairs | "
          f"{group.counts()} -> {path.name}")


# ── per-database processing ────────────────────────────────────────────────────

def process_clinvar(tsv_path, out_dir, threshold, min_partners):
    biogrid_counts = load_biogrid_partner_counts("clinvar")
    print("Processing ClinVar...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)
    for name, pkl_path in SUBSET_FILES["clinvar"].items():
        with open(pkl_path, "rb") as f:
            subset = pickle.load(f)
        table = build_arrays(grouped, subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, name, table)

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
        table = build_arrays(grouped, rare_benign_subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, "rare_benign", table)
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
    table = build_arrays(grouped, ar_pathogenic, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "ar_pathogenic", table)
    table = build_arrays(grouped, ad_pathogenic, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "ad_pathogenic", table)


def process_hgmd(tsv_path, out_dir, threshold, min_partners):
    biogrid_counts = load_biogrid_partner_counts("hgmd")
    print("Processing HGMD...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)
    with open(SUBSET_FILES["hgmd"]["hgmd"], "rb") as f:
        subset = pickle.load(f)
    table = build_arrays(grouped, subset, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "hgmd", table)

    # AR-only / AD-only disease-gene stratification of HGMD variants
    ar_uniprots, ad_uniprots = load_ar_ad_uniprots()
    if ar_uniprots is None:
        print(f"  ar_hgmd/ad_hgmd: skipped, {AR_AD_UNIPROT_FILE} not found "
              "(run build_ar_ad_gene_sets.py first)")
        return
    ar_hgmd = {(u, v, p) for (u, v, p) in subset if u in ar_uniprots}
    ad_hgmd = {(u, v, p) for (u, v, p) in subset if u in ad_uniprots}
    table = build_arrays(grouped, ar_hgmd, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "ar_hgmd", table)
    table = build_arrays(grouped, ad_hgmd, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "ad_hgmd", table)


def process_asd(tsv_path, out_dir, threshold, min_partners):
    """Fu et al. de novo ASD cases.

    Its own database (`asd_rows.csv.gz`), so every scored variant is an ASD case
    and there is no sub-group to select. This used to be carved out of the
    neurodev predictions using `neurodev/variant_subset.pkl`, which is not an ASD
    variant list at all -- it is the AlphaFold folding-budget cap
    (`interactor_count < 10`, stopping at 600 complexes), and its 290 variants
    are a strict subset of the 5,580 NeuroDev ones. The two cohorts are distinct:
    ASD is de novo autism cases from Fu et al., NeuroDev is case/control across
    four disorders.
    """
    biogrid_counts = load_biogrid_partner_counts("asd")
    print("Processing ASD (Fu et al. de novo cases)...")
    grouped = group_by_variant(load_predictions(tsv_path))
    everything = {(u, v, p) for (u, v), ps in grouped.items() for p in ps}
    table = build_arrays(grouped, everything, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "asd", table)


def process_neurodev(tsv_path, out_dir, threshold, min_partners):
    biogrid_counts = load_biogrid_partner_counts("neurodev")
    print("Processing neurodev (NDD case/control)...")
    pairs = load_predictions(tsv_path)
    grouped = group_by_variant(pairs)

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
        neurodev_out = out_dir
        if ndd_case_subset:
            table = build_arrays(grouped, ndd_case_subset, threshold, min_partners, biogrid_counts)
            save_outputs(neurodev_out, "ndd_case", table)
        if ndd_control_subset:
            table = build_arrays(grouped, ndd_control_subset, threshold, min_partners, biogrid_counts)
            save_outputs(neurodev_out, "ndd_control", table)
    else:
        print(f"  neurodev: label file not found at {NEURODEV_LABEL_FILE}")


def process_gnomad(tsv_path, out_dir, threshold, min_partners):
    biogrid_counts = load_biogrid_partner_counts("gnomad")
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
    table = build_arrays(grouped, all_gnomad_pairs, threshold, min_partners, biogrid_counts)
    save_outputs(out_dir, "gnomad", table)

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
        table = build_arrays(grouped, af_subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, name, table)
        prev_thresh = af_thresh


def process_cosmic(tsv_path, out_dir, threshold, min_partners):
    biogrid_counts = load_biogrid_partner_counts("cosmic")
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
        table = build_arrays(grouped, subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, name, table)

    # Oncogene subsets
    for name, min_rec, max_rec in bin_defs:
        onco_name = name.replace("cosmic_", "cosmic_onco_")
        subset = recurrence_subset(all_cosmic_pairs, min_rec, max_rec, "oncogene")
        if not subset:
            print(f"  {onco_name}: no variants, skipping")
            continue
        table = build_arrays(grouped, subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, onco_name, table)

    # TSG subsets
    for name, min_rec, max_rec in bin_defs:
        tsg_name = name.replace("cosmic_", "cosmic_tsg_")
        subset = recurrence_subset(all_cosmic_pairs, min_rec, max_rec, "TSG")
        if not subset:
            print(f"  {tsg_name}: no variants, skipping")
            continue
        table = build_arrays(grouped, subset, threshold, min_partners, biogrid_counts)
        save_outputs(out_dir, tsg_name, table)


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
                   choices=["clinvar", "gnomad", "hgmd", "cosmic", "neurodev", "asd"],
                   default=["clinvar", "gnomad", "hgmd", "cosmic", "neurodev", "asd"],
                   help="Databases to process (default: all)")
    p.add_argument("--pred-dir", default=str(DEFAULT_PRED_DIR),
                   help=f"Root holding {{db}}/mutpred_ppi_predictions.tsv per "
                        f"database (default: {DEFAULT_PRED_DIR}). All databases "
                        f"in one run must come from the same model.")
    args = p.parse_args()

    db_funcs = {
        "clinvar":   (process_clinvar,  "clinvar"),
        "gnomad":    (process_gnomad,   "gnomad"),
        "hgmd":      (process_hgmd,     "hgmd"),
        "cosmic":    (process_cosmic,   "cosmic"),
        "neurodev":  (process_neurodev,  "neurodev"),
        "asd":       (process_asd,       "asd"),
    }
    print(f"Prediction TSVs: {args.pred_dir}", flush=True)

    for db in args.databases:
        func, out_subdir = db_funcs[db]
        tsv_path = prediction_tsv(args.pred_dir, db)
        out_dir = os.path.join(args.output_dir, out_subdir)
        if not os.path.exists(tsv_path):
            print(f"WARNING: TSV not found: {tsv_path}, skipping {db}")
            continue
        func(tsv_path, out_dir, args.threshold, args.min_partners)

    print("\nDone.")


if __name__ == "__main__":
    main()
