#!/usr/bin/env python
"""Build a master gzip-compressed CSV of all variant-partner predictions.

Uses the all-data model's (sahni_fragoza_varchamp_all_mapped090826) predictions
across all SIX variant databases -- ClinVar, gnomAD, COSMIC, HGMD, NDD and ASD.
That is the analysis scope.

COSMIC and HGMD are licence-restricted and must be dropped from anything
DEPOSITED or shared: `--deposition` does that, leaving the four redistributable
databases. The restriction is about redistribution, not about what the analysis
should cover, so it is applied on the way out rather than by narrowing the
inputs.

Output columns:
  interactor_uniprot, variant, partner_uniprot,
  mutpredppi_score,
  is_PLP, is_BLB, is_rare_BLB, is_VUS,
  is_gnomAD, gnomAD_AF,
  is_ASD, is_NDD_case, is_NDD_control,
  training_overlap   -- True when this (interactor, variant) appears in the
                        model's training set, partner ignored. The figures
                        exclude these rows; the deposited table keeps them and
                        flags them, so either view is reproducible.
  [is_COSMIC, is_COSMIC_oncogene, is_COSMIC_TSG, cosmic_recurrence  -- if --include-cosmic]
"""
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path

import pandas as pd

# --- repo-relative path resolution (see src/paths.py) ---
from paths import VARIANT_DBS_DIR, ANNOTATIONS_DIR, ANNOTATIONS_LICENSED_DIR, DATA_ROOT, RESULTS_DIR  # noqa: E402


# ── paths ─────────────────────────────────────────────────────────────────────
_BASE = DATA_ROOT
_HOME   = _BASE / "home"
_PUB    = _BASE / "publication"
_REVDIR = RESULTS_DIR

# Where the inference actually writes, taken from the runner rather than
# re-spelled here. These used to be hardcoded under `results/variant_dbs_all_data/`
# -- a path the runner has never written to -- so every TSV silently failed to
# load and the only symptom was a KeyError on a column the empty-fallback frame
# does not carry.
from variant_db_inference.run_variant_db_inference import (  # noqa: E402
    DATASET_CONFIGS as _VDB_CONFIGS)
from analysis import training_overlap  # noqa: E402

_DEFAULT_OUT = str(_REVDIR / "master_variant_db_predictions.csv.gz")

_MASTER_DBS = ("clinvar", "gnomad", "cosmic", "hgmd", "neurodev", "asd")
# Dropped by --deposition; see the module docstring.
_LICENCE_RESTRICTED = ("cosmic", "hgmd")
def _prediction_tsv(db: str):
    """One repository's predictions, in either supported layout.

    The runner's own default is `$MUTPRED_DATA_ROOT/<db>/mutpred_ppi_predictions.tsv`,
    but notebooks/reproduce_all_figures.py passes `--out` to collect them under
    `<results>/variant_dbs_all_data/<db>_mutpred_ppi_predictions.tsv`, and that
    collected layout is also what the Zenodo deposit unpacks to. Looking only at
    the runner default meant every TSV silently failed to load for anyone who
    had not run inference in the legacy layout. Same resolution order as
    `src/build_zenodo_deposit.py::_prediction_tsv`.
    """
    collected = VARIANT_DBS_DIR / f"{db}_mutpred_ppi_predictions.tsv"
    return collected if collected.exists() else _VDB_CONFIGS[db]["default_out"]


ALL_DATA_TSV = {db: _prediction_tsv(db) for db in _MASTER_DBS}

CLINVAR_PKL = {
    "pathogenic": ANNOTATIONS_DIR / "clinvar" / "pathogenic_dirbind_variant_subset.pkl",
    "benign":     ANNOTATIONS_DIR / "clinvar" / "benign_dirbind_variant_subset.pkl",
    "vus":        ANNOTATIONS_DIR / "clinvar" / "vus_dirbind_variant_subset.pkl",
}
GNOMAD_AF_TSV      = ANNOTATIONS_DIR / "gnomad_allele_frequencies.tsv"
BENIGN_AF_TSV      = ANNOTATIONS_DIR / "benign_allele_frequencies.tsv"
RARE_BLB_AF_THRESH = 0.01
ASD_SUBSET_PKL     = ANNOTATIONS_DIR / "neurodev" / "variant_subset.pkl"
NDD_LABEL_PKL   = ANNOTATIONS_DIR / "neurodev" / "variant_label_dict.pkl"

# COSMIC (only used with --include-cosmic)
COSMIC_ONCO_TSG_PKL = ANNOTATIONS_LICENSED_DIR / "onco_tsg_dict.pkl"
COSMIC_VT_SITE_PKL  = ANNOTATIONS_LICENSED_DIR / "vt_to_tumor_site.pkl"
# `cosmic_recurrence` below is len() of that list, i.e. one entry PER OCCURRENCE (a site repeats once per sample), so len() is the recurrence -- the number of times the variant was observed, NOT the number of distinct tissues.


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


def _require_hits(label: str, n_hit: int, n_total: int,
                  min_frac: float = 0.0) -> None:
    """Fail when an annotation join matches nothing (or implausibly little).

    Every annotation here is a dict lookup keyed on a string built at the call
    site. If the key SHAPE is wrong -- tuple vs string, wrong field order, wrong
    case -- every lookup misses and the column fills with a valid-looking
    default. Nothing raises, and the artifact ships with an empty annotation.
    Checking the hit rate turns that class of bug into a failure.
    """
    if n_total == 0:
        return
    frac = n_hit / n_total
    if n_hit == 0:
        raise ValueError(
            f"{label}: 0/{n_total:,} rows matched. The annotation cache loaded but "
            f"nothing joined -- check the key shape, not the data.")
    if frac < min_frac:
        raise ValueError(
            f"{label}: only {n_hit:,}/{n_total:,} rows matched ({frac:.3%}), "
            f"below the expected {min_frac:.1%}.")
    print(f"  {label}: {n_hit:,}/{n_total:,} ({frac:.1%})")


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
    include_cosmic = args.include_cosmic and not args.deposition
    dbs = {db: p for db, p in ALL_DATA_TSV.items()
           if not (args.deposition and db in _LICENCE_RESTRICTED)}
    if args.deposition:
        print(f"deposition mode: dropping {_LICENCE_RESTRICTED}")

    print("Loading all-data-model predictions...")
    all_data = load_all_tsv(dbs, "mutpredppi_score", include_cosmic)
    print(f"  {len(all_data):,} rows")

    key_cols = ["interactor_uniprot", "variant", "partner_uniprot"]

    if all_data.empty:
        looked = "\n".join(f"    {p}" for p in dbs.values())
        raise SystemExit(
            "No prediction TSVs could be loaded, so there is nothing to build a "
            f"master table from. Looked for:\n{looked}\n"
            "  Run src/variant_db_inference/run_variant_db_inference.py first, "
            "or unpack the Zenodo results archive.")

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
        # COSMIC and HGMD are licensed, so a clean reproduction may legitimately
        # have neither cache. Absent is a WARNING and the columns stay empty;
        # present-but-matching-nothing is an ERROR, because that means the key
        # shape is wrong rather than the data being unavailable.
        onco_tsg: dict = {}
        vt_sites: dict = {}
        missing = [p for p in (COSMIC_ONCO_TSG_PKL, COSMIC_VT_SITE_PKL)
                   if not p.exists()]
        if missing:
            print("  WARNING: COSMIC annotation cache(s) not present:")
            for m in missing:
                print(f"    {m}")
            print("  COSMIC columns will be empty. This is expected without a "
                  "COSMIC licence; rebuild them with "
                  "src/data_processing/variant_databases/get_cosmic_annotations.py")
        else:
            with open(COSMIC_ONCO_TSG_PKL, "rb") as f:
                onco_tsg = pickle.load(f)
            with open(COSMIC_VT_SITE_PKL, "rb") as f:
                vt_sites = pickle.load(f)
        cosmic_caches_present = not missing and bool(vt_sites)

        # Both caches are keyed by the VARIANT as "ACCESSION MUTATION", e.g.
        # "Q15389 T227P" -- not by a (accession, variant) tuple, and `onco_tsg`
        # is class -> set-of-variants, not accession -> {oncogene, tsg}. Getting
        # either shape wrong makes every lookup miss silently: the columns fill
        # with 0/False and the only symptom is "COSMIC rows: 0".
        variant_keys_cosmic = [f"{a} {v}" for a, v in
                               zip(master["interactor_uniprot"], master["variant"])]
        onco = onco_tsg.get("oncogene", set())
        tsg = onco_tsg.get("TSG", set())
        master["cosmic_recurrence"] = [len(vt_sites.get(k, ())) for k in variant_keys_cosmic]
        master["is_COSMIC"] = master["cosmic_recurrence"] > 0
        master["is_COSMIC_oncogene"] = [k in onco for k in variant_keys_cosmic]
        master["is_COSMIC_TSG"] = [k in tsg for k in variant_keys_cosmic]
        n_cosmic = int(master["is_COSMIC"].sum())
        print(f"  COSMIC rows: {n_cosmic:,}")
        # A join that matches NOTHING is a key-shape bug, not a finding. Both of
        # these caches are keyed "ACCESSION MUTATION"; reading them as tuples, or
        # as accession -> {oncogene, tsg}, silently produced all-zero columns and
        # the only symptom was this count printing 0.
        if cosmic_caches_present:
            _require_hits("COSMIC recurrence", n_cosmic, len(master))
            _require_hits("COSMIC oncogene",
                          int(master["is_COSMIC_oncogene"].sum()), len(master))
            _require_hits("COSMIC TSG",
                          int(master["is_COSMIC_TSG"].sum()), len(master))

    # ── Training overlap ──────────────────────────────────────────────────────
    # Flag rather than drop. Every enrichment figure excludes these rows (see
    # analysis/training_overlap.py); the deposited table keeps them so a reader
    # can reproduce the filtered view, or check what the filter removed, without
    # needing the training set -- which is not redistributable.
    known = training_overlap.training_variants_or_none()
    if known is None:
        print("[warn] training table unavailable; the "
              f"{training_overlap.OVERLAP_COLUMN!r} column cannot be computed "
              "and is omitted", flush=True)
    else:
        keys = zip(master["interactor_uniprot"].astype(str),
                   master["variant"].astype(str))
        master[training_overlap.OVERLAP_COLUMN] = [k in known for k in keys]
        n = int(master[training_overlap.OVERLAP_COLUMN].sum())
        print(f"\n{training_overlap.OVERLAP_COLUMN}: {n:,} of {len(master):,} rows "
              f"({100 * n / len(master):.2f}%) are variants seen in "
              f"{training_overlap.TRAINING_DATASET}", flush=True)

    # ── Output ────────────────────────────────────────────────────────────────
    # The deposition file is a DIFFERENT artifact, not a replacement: writing
    # both to one path means whichever ran last is whatever you have.
    out_path = Path(args.output)
    if args.deposition and args.output == _DEFAULT_OUT:
        out_path = out_path.with_name(
            out_path.name.replace(".csv.gz", "_unrestricted.csv.gz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False, compression="gzip")
    print(f"\nWrote {len(master):,} rows → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=_DEFAULT_OUT,
                   help="Output gzip CSV path")
    p.add_argument("--include-cosmic", action=argparse.BooleanOptionalAction, default=True,
                   help="Include COSMIC rows/annotations (default: True). "
                        "Check the COSMIC licence before sharing the output.")
    p.add_argument("--deposition", action="store_true",
                   help=f"Drop the licence-restricted databases {_LICENCE_RESTRICTED} "
                        "for a redistributable file. Use this for anything deposited.")
    main(p.parse_args())
