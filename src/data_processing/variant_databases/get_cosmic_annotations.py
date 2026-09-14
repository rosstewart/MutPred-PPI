#!/usr/bin/env python
"""Derive the COSMIC per-variant annotations from the Cancer Mutation Census.

Migrates two notebooks that had no counterpart in this repo. They read the same
file, applied the same filters and the same key construction, and differed only
in what they accumulated per row, so they collapse to one script with two
stages -- the same consolidation `map_clinvar.py` does:

    get_cosmic_recurrence.ipynb -> --stage recurrence
    get_cosmic_onco_TSG.ipynb   -> --stage onco-tsg

`--stage both` (the default) runs one pass over the CMC and fills both
accumulators; they are independent, so this is identical to running the two
notebooks separately, minus a second 5-minute read of a 291 MB TSV.

WHY THIS MATTERS
----------------
`recurrence_dict.pkl` is a PREREQUISITE of the already-migrated `map_cosmic.py`
(it is that script's `--gene-symbol-to-uniprot` companion output, and the thing
that decides which COSMIC variants get triplets at all), and `onco_tsg_dict.pkl`
is read by `analysis/cosmic_onco_tsg_stat_test.py`. Both were produced outside
version control until now.

TRANSCRIPTION, NOT REDESIGN
---------------------------
Filters, cutoffs and conventions are carried over verbatim; only I/O, paths and
the CLI are rewritten. Conventions that look like bugs:

  * Recurrence is binned from COSMIC_SAMPLE_MUTATED with thresholds
    2/4/8/16/32/64 into levels 0-6. That is intended.
  * RECURRENCE MAX -- FIXED, with the old behaviour still reachable.
    The notebook's "keep the largest value per variant" test was
    `if key in recurrence_dict and n_recurrence < recurrence_dict[key]: continue`,
    comparing a RAW SAMPLE COUNT against the previously stored 0-6 SCORE. Counts
    are almost always larger than a score, so the guard rarely fired and the
    effective rule was "last row wins", not "keep the max". Measured: 2,456 of
    the 23,464 variants with more than one CMC row ended up BELOW their true
    maximum. `map_cosmic.py` already computed a true max, so the repo held two
    inconsistent definitions of COSMIC recurrence.
    This script now computes the true max by default. `--legacy-recurrence`
    reproduces the published `recurrence_dict.pkl` bit-for-bit.
  * EMPTY UNIPROT IDS -- FIXED. `gene_symbol_to_uniprot` maps some genes to an
    empty string (HGNC listed no `uniprot_ids`). The notebook still built a key
    from it, so 76 genes pooled 8,043 variants under the meaningless key
    ' <variant>' and put '' into `proteins_with_recurrence_data`. Those rows are
    now skipped and counted. `--keep-empty-uniprot` restores the old behaviour.
  * Variant keys are '<uniprot> <variant>' with 1-BASED positions, exactly as
    COSMIC writes them ('p.R175H' -> 'R175H'). No re-basing happens here; the
    0-based conversion belongs to sequence indexing in `map_cosmic.py`.
  * Rows whose `Mutation AA` contains '*' (frameshift/nonsense, e.g.
    'p.V1163Gfs*3') are dropped, as are genes absent from
    `gene_symbol_to_uniprot.pkl`.
  * ONC_TSG is matched by EXACT equality against 'oncogene' and 'TSG', so
    combined labels such as 'oncogene, TSG' are silently dropped. Verbatim.
  * The onco/TSG sets are NOT restricted to variants present in
    recurrence_dict; `map_cosmic.py` adds such a restriction, which is why its
    onco_tsg_dict is smaller than the published one.

Inputs:
    --cmc-file                CancerMutationCensus_AllData_v*.tsv.gz (licensed)
    --gene-symbol-to-uniprot  gene_symbol_to_uniprot.pkl, built by the HGNC pass
                              of map_cosmic.py

Outputs (--stage recurrence):
    recurrence_dict.pkl              {'<uniprot> <variant>': 0-6}
    proteins_with_recurrence_data.pkl  {uniprot}
Outputs (--stage onco-tsg):
    onco_tsg_dict.pkl                {'oncogene': {...}, 'TSG': {...}}
    proteins_with_onc_tsg.pkl        {uniprot}
Outputs (--stage occurrences):
    vt_to_tumor_site.pkl             {'<uniprot> <variant>': [site, ...]}
                                     ONE ENTRY PER OCCURRENCE, so len() is the
                                     recurrence -- NOT a set of distinct tissues
    vt_to_occurrence_count.pkl       {'<uniprot> <variant>': n_occurrences}

LICENSING: COSMIC is licensed. Only derived scores and identifier mappings are
written; no sample-level or raw mutation records leave this script.

Usage:
    python get_cosmic_annotations.py --stage both \\
        --cmc-file $MUTPRED_DATA_ROOT/cosmic_mutations/CancerMutationCensus_AllData_v101_GRCh37.tsv.gz \\
        --gene-symbol-to-uniprot $MUTPRED_DATA_ROOT/cosmic_mutations/gene_symbol_to_uniprot.pkl \\
        --output-dir ./scratch_cosmic_annotations
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

# Recurrence bin edges on COSMIC_SAMPLE_MUTATED (0-6 ordinal scale).
LOW_RECURRENCE_THRESHOLD = 2
HIGH_RECURRENCE_THRESHOLD = 4
HIGHER_RECURRENCE_THRESHOLD = 8
HIGHEST_RECURRENCE_THRESHOLD = 16
SUPER_HIGHEST_RECURRENCE_THRESHOLD = 32
SUPER_DUPER_HIGHEST_RECURRENCE_THRESHOLD = 64


def bin_recurrence(n_recurrence):
    """COSMIC_SAMPLE_MUTATED -> 0-6, transcribed from the notebook's nesting."""
    if n_recurrence < LOW_RECURRENCE_THRESHOLD:
        return 0
    if n_recurrence >= HIGH_RECURRENCE_THRESHOLD:
        if n_recurrence >= HIGHER_RECURRENCE_THRESHOLD:
            if n_recurrence >= HIGHEST_RECURRENCE_THRESHOLD:
                if n_recurrence >= SUPER_HIGHEST_RECURRENCE_THRESHOLD:
                    return (6 if n_recurrence >= SUPER_DUPER_HIGHEST_RECURRENCE_THRESHOLD
                            else 5)
                return 4
            return 3
        return 2
    assert (n_recurrence >= LOW_RECURRENCE_THRESHOLD
            and n_recurrence < HIGH_RECURRENCE_THRESHOLD)
    return 1


def load_cmc_missense(cmc_file):
    import pandas as pd
    print(f"reading {cmc_file} ...", flush=True)
    df = pd.read_csv(cmc_file, sep="\t", compression="gzip", low_memory=False)
    df = df[df["Mutation Description AA"] == "Substitution - Missense"]
    print(f"  {len(df)} missense records", flush=True)
    return df


def accumulate(df, gene_symbol_to_uniprot, want_recurrence, want_onco_tsg,
               *, legacy_recurrence=False, keep_empty_uniprot=False):
    """Single pass over the CMC rows filling the two independent accumulators."""
    n_empty_uniprot = 0
    recurrence_dict: dict = {}
    oncos, tsgs = [], []

    for _, row in df.iterrows():
        mut = row["Mutation AA"]
        gene = row["GENE_NAME"]
        if "*" in mut or gene not in gene_symbol_to_uniprot:  # e.g. p.V1163Gfs*3
            continue
        uniprot = gene_symbol_to_uniprot[gene]
        if not uniprot and not keep_empty_uniprot:
            n_empty_uniprot += 1
            continue
        mut = mut[2:]  # remove 'p.'
        key = f"{uniprot} {mut}"

        if want_recurrence:
            score = bin_recurrence(row["COSMIC_SAMPLE_MUTATED"])
            if legacy_recurrence:
                # Published behaviour: raw count vs stored score, i.e. last wins.
                if not (key in recurrence_dict
                        and row["COSMIC_SAMPLE_MUTATED"] < recurrence_dict[key]):
                    recurrence_dict[key] = score
            elif key not in recurrence_dict or score > recurrence_dict[key]:
                recurrence_dict[key] = score

        if want_onco_tsg:
            onc_tsg = row["ONC_TSG"]
            if onc_tsg == "oncogene":
                oncos.append(key)
            elif onc_tsg == "TSG":
                tsgs.append(key)

    if n_empty_uniprot:
        print(f"  skipped {n_empty_uniprot} rows whose gene maps to an empty "
              f"UniProt id", flush=True)
    return recurrence_dict, {"oncogene": set(oncos), "TSG": set(tsgs)}


def build_occurrences(genome_screens_file: str, classification_file: str,
                      gene_symbol_to_uniprot: dict) -> dict:
    """`{'<uniprot> <variant>': [primary_site, ...]}` -- ONE ENTRY PER OCCURRENCE.

    RECURRENCE IS THE NUMBER OF OCCURRENCES OF A VARIANT, not the number of
    distinct tissues. The list therefore repeats a site once per sample carrying
    that variant, and `len(sites)` is the recurrence. (`P35222 S45F`: 2,884
    entries across 17 distinct sites -> recurrence 2,884, not 17.) The list is
    kept rather than a bare count only because the sites themselves are used for
    tumour-type breakdowns; `--stage occurrences` also writes the plain counts.

    The historical name for this file is `vt_to_tumor_site.pkl`, which describes
    its VALUES and not the quantity read off it -- the misreading that name
    invites is exactly why the count is spelled out here.

    Built by joining COSMIC's per-sample mutation table to its phenotype
    classification: GenomeScreensMutant gives one row per sample-variant, and
    COSMIC_PHENOTYPE_ID resolves to PRIMARY_SITE.
    """
    import pandas as pd

    print(f"Reading classification {classification_file} ...", flush=True)
    cls = pd.read_csv(classification_file, sep="\t", compression="gzip",
                      low_memory=False,
                      usecols=["COSMIC_PHENOTYPE_ID", "PRIMARY_SITE"])
    pheno_to_site = dict(zip(cls["COSMIC_PHENOTYPE_ID"], cls["PRIMARY_SITE"]))
    print(f"  {len(pheno_to_site):,} phenotype -> primary site", flush=True)

    print(f"Reading per-sample mutations {genome_screens_file} ...", flush=True)
    out: dict[str, list] = {}
    n_rows = n_kept = 0
    for chunk in pd.read_csv(genome_screens_file, sep="\t", compression="gzip",
                             low_memory=False, chunksize=1_000_000,
                             usecols=["GENE_SYMBOL", "MUTATION_AA",
                                      "COSMIC_PHENOTYPE_ID"]):
        for gene, mut_aa, pheno in zip(chunk["GENE_SYMBOL"], chunk["MUTATION_AA"],
                                       chunk["COSMIC_PHENOTYPE_ID"]):
            n_rows += 1
            if not isinstance(mut_aa, str) or "*" in mut_aa:
                continue
            uid = gene_symbol_to_uniprot.get(gene)
            if not uid:
                continue
            # `p.A123V` -> `A123V`; positions are 1-based, as everywhere else.
            out.setdefault(f"{uid} {mut_aa[2:]}", []).append(
                pheno_to_site.get(pheno, "unknown"))
            n_kept += 1
        print(f"    {n_rows:,} rows read, {n_kept:,} occurrences kept", flush=True)
    print(f"  {len(out):,} unique variants, {n_kept:,} total occurrences", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",
                    choices=("recurrence", "onco-tsg", "occurrences", "both", "all"),
                    default="both",
                    help="'both' = recurrence + onco-tsg (CMC only). "
                         "'occurrences' rebuilds vt_to_tumor_site.pkl and needs "
                         "the per-sample files. 'all' runs everything.")
    ap.add_argument("--genome-screens-file", default=None,
                    help="Cosmic_GenomeScreensMutant_Missense_v*.tsv.gz "
                         "(licensed; required for --stage occurrences/all)")
    ap.add_argument("--classification-file", default=None,
                    help="Cosmic_Classification_v*.tsv.gz "
                         "(licensed; required for --stage occurrences/all)")
    ap.add_argument("--cmc-file", required=True,
                    help="COSMIC CancerMutationCensus AllData TSV.gz (licensed)")
    ap.add_argument("--gene-symbol-to-uniprot", required=True,
                    help="gene_symbol_to_uniprot.pkl (HGNC pass of map_cosmic.py)")
    ap.add_argument("--output-dir", default="./scratch_cosmic_annotations")
    ap.add_argument("--legacy-recurrence", action="store_true",
                    help="reproduce the published dict, whose 'keep the max' test "
                         "compared a raw count to a stored score and so kept the "
                         "LAST row rather than the largest")
    ap.add_argument("--keep-empty-uniprot", action="store_true",
                    help="keep rows whose gene maps to an empty UniProt id, "
                         "pooling them under the key ' <variant>' as the notebook did")
    args = ap.parse_args()

    want_recurrence = args.stage in ("recurrence", "both", "all")
    want_onco_tsg = args.stage in ("onco-tsg", "both", "all")
    want_occurrences = args.stage in ("occurrences", "all")
    if want_occurrences and not (args.genome_screens_file and args.classification_file):
        ap.error("--stage occurrences/all needs --genome-screens-file and "
                 "--classification-file")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.gene_symbol_to_uniprot, "rb") as f:
        gene_symbol_to_uniprot = pickle.load(f)
    print(f"{len(gene_symbol_to_uniprot)} gene->UniProt mappings", flush=True)

    df = load_cmc_missense(args.cmc_file)
    if want_onco_tsg:
        print(f"  ONC_TSG values: {df['ONC_TSG'].value_counts().to_dict()}", flush=True)

    recurrence_dict, onco_tsg_dict = accumulate(
        df, gene_symbol_to_uniprot, want_recurrence, want_onco_tsg,
        legacy_recurrence=args.legacy_recurrence,
        keep_empty_uniprot=args.keep_empty_uniprot)

    if want_recurrence:
        proteins_with_recurrence_data = {k.split(" ")[0] for k in recurrence_dict}
        print(f"recurrence: {len(recurrence_dict)} variants, "
              f"{len(proteins_with_recurrence_data)} proteins", flush=True)
        with open(os.path.join(args.output_dir, "recurrence_dict.pkl"), "wb") as f:
            pickle.dump(recurrence_dict, f)
        with open(os.path.join(args.output_dir,
                               "proteins_with_recurrence_data.pkl"), "wb") as f:
            pickle.dump(proteins_with_recurrence_data, f)

    if want_onco_tsg:
        proteins_with_onc_tsg = set()
        for key in onco_tsg_dict["oncogene"]:
            proteins_with_onc_tsg.add(key.split(" ")[0])
        for key in onco_tsg_dict["TSG"]:
            proteins_with_onc_tsg.add(key.split(" ")[0])
        print(f"onco/TSG: {len(onco_tsg_dict['oncogene'])} oncogene, "
              f"{len(onco_tsg_dict['TSG'])} TSG variants, "
              f"{len(proteins_with_onc_tsg)} proteins", flush=True)
        with open(os.path.join(args.output_dir, "onco_tsg_dict.pkl"), "wb") as f:
            pickle.dump(onco_tsg_dict, f)
        with open(os.path.join(args.output_dir,
                               "proteins_with_onc_tsg.pkl"), "wb") as f:
            pickle.dump(proteins_with_onc_tsg, f)

    if want_occurrences:
        occ = build_occurrences(args.genome_screens_file, args.classification_file,
                                gene_symbol_to_uniprot)
        with open(os.path.join(args.output_dir, "vt_to_tumor_site.pkl"), "wb") as f:
            pickle.dump(occ, f)
        # The quantity consumers actually read off the file above, stored plainly
        # so nothing has to re-derive it with len() and call it a site count.
        counts = {k: len(v) for k, v in occ.items()}
        with open(os.path.join(args.output_dir,
                               "vt_to_occurrence_count.pkl"), "wb") as f:
            pickle.dump(counts, f)
        top = max(counts.items(), key=lambda kv: kv[1]) if counts else ("-", 0)
        print(f"  most recurrent variant: {top[0]} with {top[1]:,} occurrences",
              flush=True)

    print(f"outputs written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
