#!/usr/bin/env python
"""Build a master gzip-compressed CSV of all variant-partner predictions.

Uses the all-data model's (sahni_fragoza_varchamp_all_mapped090826) predictions across
ClinVar, gnomAD, and autism/NDD datasets. HGMD is excluded (commercial
license). COSMIC is excluded by default due to redistribution restrictions;
enable with --include-cosmic if you have verified your use is compliant.

Output columns:
  interactor_uniprot, variant, partner_uniprot,
  mutpredppi_score,
  is_PLP, is_BLB, is_rare_BLB, is_VUS,
  is_gnomAD, gnomAD_AF,
  is_ASD, is_NDD_case, is_NDD_control
  [is_COSMIC, is_COSMIC_oncogene, is_COSMIC_TSG, cosmic_recurrence  -- if --include-cosmic]
"""
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
import sys as _sys
from pathlib import Path as _Path
from paths import ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT  # noqa: E402


# ── paths ─────────────────────────────────────────────────────────────────────
_BASE = DATA_ROOT
_HOME   = _BASE / "home"
_PUB    = _BASE / "publication"
_REVDIR = _PUB / "results"

ALL_DATA_TSV = {
    "clinvar": _REVDIR / "variant_dbs_all_data" / "clinvar_mutpred_ppi_predictions.tsv",
    "gnomad":  _REVDIR / "variant_dbs_all_data" / "gnomad_mutpred_ppi_predictions.tsv",
    "cosmic":  _REVDIR / "variant_dbs_all_data" / "cosmic_mutpred_ppi_predictions.tsv",
    "autism":  _REVDIR / "variant_dbs_all_data" / "autism_mutpred_ppi_predictions.tsv",
}

CLINVAR_PKL = {
    "pathogenic": ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl",
    "benign":     ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl",
    "vus":        ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl",
}
GNOMAD_AF_TSV      = ANNOTATIONS_DIR / "gnomad_allele_frequencies.tsv"
BENIGN_AF_TSV      = ANNOTATIONS_DIR / "benign_allele_frequencies.tsv"
RARE_BLB_AF_THRESH = 0.01
ASD_SUBSET_PKL     = ANNOTATIONS_DIR / "autism" / "variant_subset.pkl"
NDD_LABEL_PKL   = ANNOTATIONS_DIR / "autism" / "variant_label_dict.pkl"

# COSMIC (only used with --include-cosmic)
COSMIC_ONCO_TSG_PKL = ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl"
COSMIC_VT_SITE_PKL  = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"


def load_tsv(path: Path) -> pd.DataFrame:
    """Read a prediction TSV into explicit interactor / partner / variant columns.

    Prefers explicit `interactor` and `partner` columns. Falls back to splitting
    the legacy `complex_id` composite for TSVs written before the schema change;
    that branch exists only to keep already-published artifacts readable and
    should go once they are regenerated.
    """
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return pd.DataFrame(columns=["interactor_uniprot", "variant", "partner_uniprot", "score"])
    df = pd.read_csv(path, sep="\t")

    if {"interactor", "partner"}.issubset(df.columns):
        df["interactor_uniprot"] = df["interactor"]
        df["partner_uniprot"] = df["partner"]
        # The explicit schema names the column `mutation`; the legacy one
        # `variant`. Downstream here expects `variant`.
        if "variant" not in df.columns and "mutation" in df.columns:
            df["variant"] = df["mutation"]
    else:
        # `complex_id` is '{interactor}_{partner}'. Split on the FIRST underscore:
        # UniProt accessions never contain one, so it is the unambiguous
        # delimiter even for isoforms like 'O14787-2', whose separator is '-'.
        # (A previous comment here described splitting on the last underscore for
        # isoforms; the code never did that, and doing so would be wrong.)
        split = df["complex_id"].str.split("_", n=1, expand=True)
        df["interactor_uniprot"] = split[0]
        df["partner_uniprot"] = split[1]
        df = df.drop(columns=["complex_id"])

    return df[["interactor_uniprot", "variant", "partner_uniprot", "score"]]


def load_all_tsv(tsv_dict: dict, score_col: str, include_cosmic: bool) -> pd.DataFrame:
    frames = []
    for db, path in tsv_dict.items():
        if db == "cosmic" and not include_cosmic:
            continue
        df = load_tsv(path)
        if len(df) == 0:
            continue
        df["_source"] = db
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["interactor_uniprot", "variant", "partner_uniprot", score_col])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"score": score_col})
    return out


def main(args):
    include_cosmic = args.include_cosmic

    print("Loading all-data-model predictions...")
    all_data = load_all_tsv(ALL_DATA_TSV, "mutpredppi_score", include_cosmic)
    print(f"  {len(all_data):,} rows")

    key_cols = ["interactor_uniprot", "variant", "partner_uniprot"]

    master = all_data.drop(columns=["_source"])
    master = master.drop_duplicates(subset=key_cols)
    print(f"  {len(master):,} unique (interactor, variant, partner) rows")

    # ── ClinVar annotations ───────────────────────────────────────────────────
    print("Loading ClinVar annotations...")
    plp_set, blb_set, vus_set = set(), set(), set()
    for label, pkl_path in CLINVAR_PKL.items():
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                s = pickle.load(f)
            if label == "pathogenic": plp_set = s
            elif label == "benign":   blb_set = s
            elif label == "vus":      vus_set = s
        else:
            print(f"  [warn] {pkl_path.name} not found")

    keys = list(zip(master["interactor_uniprot"], master["variant"], master["partner_uniprot"]))
    master["is_PLP"] = [k in plp_set for k in keys]
    master["is_BLB"] = [k in blb_set for k in keys]
    master["is_VUS"] = [k in vus_set for k in keys]

    # Rare BLB: ClinVar benign with gnomAD AF <= RARE_BLB_AF_THRESH
    benign_af: dict[str, float] = {}
    if BENIGN_AF_TSV.exists():
        with open(BENIGN_AF_TSV) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    benign_af[parts[0]] = float(parts[1])
    interactor_variant_keys = master["interactor_uniprot"] + " " + master["variant"]
    master["is_rare_BLB"] = master["is_BLB"] & (
        interactor_variant_keys.map(lambda k: benign_af.get(k, 1.0) <= RARE_BLB_AF_THRESH)
    )
    print(f"  PLP={master['is_PLP'].sum()}, BLB={master['is_BLB'].sum()}, "
          f"rare_BLB={master['is_rare_BLB'].sum()}, VUS={master['is_VUS'].sum()}")

    # ── gnomAD AF ─────────────────────────────────────────────────────────────
    print("Loading gnomAD allele frequencies...")
    af_dict: dict[str, float] = {}
    if GNOMAD_AF_TSV.exists():
        with open(GNOMAD_AF_TSV) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    af_dict[parts[0]] = float(parts[1])
    else:
        print(f"  [warn] {GNOMAD_AF_TSV} not found")
    gnomad_keys = master["interactor_uniprot"] + " " + master["variant"]
    master["is_gnomAD"] = gnomad_keys.isin(af_dict)
    master["gnomAD_AF"] = gnomad_keys.map(af_dict)
    print(f"  {master['is_gnomAD'].sum():,} rows with gnomAD AF")

    # ── ASD / NDD annotations ─────────────────────────────────────────────────
    print("Loading ASD / NDD annotations...")
    asd_set: set = set()
    if ASD_SUBSET_PKL.exists():
        with open(ASD_SUBSET_PKL, "rb") as f:
            asd_set = pickle.load(f)

    ndd_dict: dict[str, int] = {}
    if NDD_LABEL_PKL.exists():
        with open(NDD_LABEL_PKL, "rb") as f:
            ndd_dict = pickle.load(f)

    master["is_ASD"] = [k in asd_set for k in keys]
    ndd_labels = gnomad_keys.map(ndd_dict)  # reuse interactor+variant key
    master["is_NDD_case"]    = ndd_labels == 1
    master["is_NDD_control"] = ndd_labels == 0
    print(f"  ASD={master['is_ASD'].sum()}, NDD_case={master['is_NDD_case'].sum()}, "
          f"NDD_control={master['is_NDD_control'].sum()}")

    # ── COSMIC (optional) ─────────────────────────────────────────────────────
    if include_cosmic:
        print("Loading COSMIC annotations...")
        onco_tsg: dict = {}
        vt_sites: dict = {}
        if COSMIC_ONCO_TSG_PKL.exists():
            with open(COSMIC_ONCO_TSG_PKL, "rb") as f:
                onco_tsg = pickle.load(f)
        if COSMIC_VT_SITE_PKL.exists():
            with open(COSMIC_VT_SITE_PKL, "rb") as f:
                vt_sites = pickle.load(f)

        variant_keys_cosmic = list(zip(master["interactor_uniprot"], master["variant"]))
        master["cosmic_recurrence"] = [len(vt_sites.get(k, [])) for k in variant_keys_cosmic]
        master["is_COSMIC"] = master["cosmic_recurrence"] > 0
        master["is_COSMIC_oncogene"] = [bool(onco_tsg.get(k[0], {}).get("oncogene", False)) for k in variant_keys_cosmic]
        master["is_COSMIC_TSG"]      = [bool(onco_tsg.get(k[0], {}).get("tsg", False)) for k in variant_keys_cosmic]
        print(f"  COSMIC rows: {master['is_COSMIC'].sum()}")

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {len(master):,} rows → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(_REVDIR / "master_variant_db_predictions.csv.gz"),
                   help="Output gzip CSV path")
    p.add_argument("--include-cosmic", action="store_true",
                   help="Include COSMIC rows/annotations (check COSMIC license before sharing)")
    main(p.parse_args())
