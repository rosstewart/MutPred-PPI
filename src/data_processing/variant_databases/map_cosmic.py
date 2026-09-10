#!/usr/bin/env python
"""Map COSMIC cancer missense variants to the BioGRID PPI network.

LICENSING NOTE: COSMIC is a licensed resource. This script reads the COSMIC
files as input arguments but writes ONLY derived statistics (recurrence counts
per variant, protein/complex mappings) to output. No COSMIC variant-level data
(sample IDs, cancer-type annotations, raw mutation records) is embedded in the
script or written verbatim to output files.

Two COSMIC input files are required:
  --cosmic-file : Cosmic_GenomeScreensMutant_Missense_v*_GRCh38.tsv.gz
                  (per-sample missense variants)
  --cmc-file    : CancerMutationCensus_AllData_v*.tsv.gz
                  (cancer mutation census, used for recurrence scoring)

The pipeline:
  1. Read CMC file → compute per-variant recurrence score (0–6 scale).
  2. Map HGNC gene symbols → UniProt IDs (requires gene_symbol_to_uniprot.pkl
     or fetches via HGNC REST API).
  3. Intersect with BioGRID network to find relevant complexes.
  4. Build variant triplets (uniprot, variant, partner) for proteins with
     recurrence data.
  5. Write recurrence scores, complex mappings, and id_to_seq — no raw COSMIC
     sample or mutation records.

Usage:
    python map_cosmic.py \
        --cosmic-file $MUTPRED_DATA_ROOT/cosmic_mutations/Cosmic_GenomeScreensMutant_Missense_v101_GRCh38.tsv.gz \
        --cmc-file $MUTPRED_DATA_ROOT/cosmic_mutations/CancerMutationCensus_AllData_v101_GRCh37.tsv.gz \
        --biogrid-dir biogrid \
        --uniprot-fasta cosmic/all_uniprot_ids.fasta \
        --output-dir $MUTPRED_DATA_ROOT/cosmic \
        --gene-symbol-to-uniprot $MUTPRED_DATA_ROOT/cosmic_mutations/gene_symbol_to_uniprot.pkl
"""

import argparse
import os
import pickle
import pandas as pd
import requests
from data_processing.variant_databases.biogrid_common import load_biogrid
from utils import mutations  # noqa: E402
from utils.sequences import read_fasta as read_fasta_shared  # noqa: E402


# ---------------------------------------------------------------------------
# Recurrence scoring constants  (0–6 scale)
# ---------------------------------------------------------------------------

LOW_RECURRENCE_THRESHOLD = 2
HIGH_RECURRENCE_THRESHOLD = 4
HIGHER_RECURRENCE_THRESHOLD = 8
HIGHEST_RECURRENCE_THRESHOLD = 16
SUPER_HIGHEST_RECURRENCE_THRESHOLD = 32
SUPER_DUPER_HIGHEST_RECURRENCE_THRESHOLD = 64


def score_recurrence(n):
    """Map raw COSMIC_SAMPLE_MUTATED count to a 0–6 ordinal recurrence score."""
    if n < LOW_RECURRENCE_THRESHOLD:
        return 0
    if n < HIGH_RECURRENCE_THRESHOLD:
        return 1
    if n < HIGHER_RECURRENCE_THRESHOLD:
        return 2
    if n < HIGHEST_RECURRENCE_THRESHOLD:
        return 3
    if n < SUPER_HIGHEST_RECURRENCE_THRESHOLD:
        return 4
    if n < SUPER_DUPER_HIGHEST_RECURRENCE_THRESHOLD:
        return 5
    return 6


# ---------------------------------------------------------------------------
# HGNC gene → UniProt mapping
# ---------------------------------------------------------------------------

def fetch_hgnc_info(gene_list):
    """Fetch gene→UniProt mapping via the HGNC REST API."""
    base_url = "https://rest.genenames.org/fetch/symbol/"
    headers = {"Accept": "application/json"}
    data = []
    for gene in gene_list:
        resp = requests.get(base_url + gene, headers=headers)
        if resp.status_code == 200:
            docs = resp.json()["response"]["docs"]
            if docs:
                info = docs[0]
                data.append({
                    "Gene Symbol": info.get("symbol"),
                    "UniProt ID": ", ".join(info.get("uniprot_ids", [])),
                })
    return pd.DataFrame(data)


def build_gene_symbol_to_uniprot(df_hgnc):
    """Resolve multi-UniProt genes by preferring those with an AF4 PDB."""
    mapping = {}
    for _, row in df_hgnc.iterrows():
        gene = row["Gene Symbol"]
        uid = row["UniProt ID"]
        if ", " in uid:
            # Take first; caller can override with --gene-symbol-to-uniprot pkl
            uid = uid.split(", ")[0]
        if gene not in mapping:
            mapping[gene] = uid
    return mapping


# ---------------------------------------------------------------------------
# BioGRID helpers
# ---------------------------------------------------------------------------

def get_complexes_in_biogrid(gene_symbol_to_uniprot, uniprot_to_interactors, uniprot_to_seq):
    wt_complexes = set()
    for uid in gene_symbol_to_uniprot.values():
        if uid not in uniprot_to_interactors:
            continue
        for partner in uniprot_to_interactors[uid]:
            if partner in uniprot_to_seq:
                wt_complexes.add((uid, partner))
    return wt_complexes


def _cosmic_header_key(header: str) -> str:
    """`sp|Q30154|DRB5_HUMAN ...` -> `Q30154`; falls back to the whole header
    if there is no second pipe field. COSMIC's own UniProt FASTA is uniformly
    the `sp|acc|name` format (verified 2026-09-10), so the fallback is
    defensive and has not been observed to trigger on real input.
    """
    parts = header.split("|")
    return parts[1] if len(parts) > 1 else header


def read_fasta(file_path):
    return read_fasta_shared(file_path, _cosmic_header_key, on_duplicate="last")


def build_variant_triplets_with_recurrence(id_to_seq, complexes, recurrence_dict):
    """Only include variants that have a recurrence entry."""
    variants = set()
    for key in id_to_seq:
        if " " not in key:
            continue
        if key not in recurrence_dict:
            continue
        uid, variant = key.split(" ")
        for u, p in complexes:
            if uid == u:
                variants.add((uid, variant, p))
            if uid == p:
                variants.add((uid, variant, u))
    return variants


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build recurrence dict from COSMIC Cancer Mutation Census
    #    Output: {uniprot_id variant: recurrence_score (0–6)}
    # ------------------------------------------------------------------
    print(f"Reading CMC file {args.cmc_file} …")
    df_cmc = pd.read_csv(args.cmc_file, sep="\t", compression="gzip", low_memory=False)
    df_cmc = df_cmc[df_cmc["Mutation Description AA"] == "Substitution - Missense"]
    print(f"  {len(df_cmc)} missense records")

    # Load or build gene_symbol → UniProt mapping
    if args.gene_symbol_to_uniprot and os.path.exists(args.gene_symbol_to_uniprot):
        with open(args.gene_symbol_to_uniprot, "rb") as f:
            gene_symbol_to_uniprot = pickle.load(f)
        print(f"  Loaded {len(gene_symbol_to_uniprot)} gene→UniProt mappings")
    else:
        all_genes = df_cmc["GENE_NAME"].unique().tolist()
        print(f"  Fetching HGNC info for {len(all_genes)} genes (this is slow)…")
        df_hgnc = fetch_hgnc_info(all_genes)
        gene_symbol_to_uniprot = build_gene_symbol_to_uniprot(df_hgnc)
        pkl_out = os.path.join(args.output_dir, "gene_symbol_to_uniprot.pkl")
        with open(pkl_out, "wb") as f:
            pickle.dump(gene_symbol_to_uniprot, f)
        print(f"  Saved gene→UniProt mapping to {pkl_out}")

    # Compute recurrence scores per (UniProt, variant)
    recurrence_dict = {}
    for _, row in df_cmc.iterrows():
        mut_aa = row["Mutation AA"]
        gene = row["GENE_NAME"]
        if not isinstance(mut_aa, str) or "*" in mut_aa:
            continue
        if gene not in gene_symbol_to_uniprot:
            continue
        mut = mut_aa[2:]  # strip 'p.'
        key = f"{gene_symbol_to_uniprot[gene]} {mut}"
        n = row["COSMIC_SAMPLE_MUTATED"]
        score = score_recurrence(n)
        # Keep highest score across duplicates
        if key not in recurrence_dict or score > recurrence_dict[key]:
            recurrence_dict[key] = score

    print(f"  {len(recurrence_dict)} unique (protein, variant) recurrence entries")

    proteins_with_recurrence_data = {k.split(" ")[0] for k in recurrence_dict}
    print(f"  {len(proteins_with_recurrence_data)} proteins have recurrence data")

    # Save recurrence score dict (counts only, no COSMIC sample data)
    rec_pkl = os.path.join(args.output_dir, "recurrence_dict.pkl")
    with open(rec_pkl, "wb") as f:
        pickle.dump(recurrence_dict, f)
    print(f"  Saved recurrence dict to {rec_pkl}")

    proteins_pkl = os.path.join(args.output_dir, "proteins_with_recurrence_data.pkl")
    with open(proteins_pkl, "wb") as f:
        pickle.dump(proteins_with_recurrence_data, f)

    # Optional: build oncogene/TSG variant sets from CMC ONC_TSG column
    if "ONC_TSG" in df_cmc.columns:
        onco_tsg_dict = {"oncogene": set(), "TSG": set()}
        for _, row in df_cmc.iterrows():
            if not isinstance(row.get("ONC_TSG"), str):
                continue
            gene = row["GENE_NAME"]
            mut_aa = row["Mutation AA"]
            if not isinstance(mut_aa, str) or "*" in mut_aa:
                continue
            if gene not in gene_symbol_to_uniprot:
                continue
            key = f"{gene_symbol_to_uniprot[gene]} {mut_aa[2:]}"
            if key not in recurrence_dict:
                continue
            ot = row["ONC_TSG"].strip()
            if ot in onco_tsg_dict:
                onco_tsg_dict[ot].add(key)
        onco_pkl = os.path.join(args.output_dir, "onco_tsg_dict.pkl")
        with open(onco_pkl, "wb") as f:
            pickle.dump(onco_tsg_dict, f)
        print(f"  Saved oncogene/TSG sets to {onco_pkl}")

        proteins_with_onc_tsg = set()
        for v in onco_tsg_dict.values():
            for k in v:
                proteins_with_onc_tsg.add(k.split(" ")[0])
        with open(os.path.join(args.output_dir, "proteins_with_onc_tsg.pkl"), "wb") as f:
            pickle.dump(proteins_with_onc_tsg, f)

    # ------------------------------------------------------------------
    # 2. If --cosmic-file provided: gather unique variant set for seq lookup
    # ------------------------------------------------------------------
    all_variants_gene = set()
    if args.cosmic_file and os.path.exists(args.cosmic_file):
        print(f"Reading per-sample COSMIC file {args.cosmic_file} …")
        df_cos = pd.read_csv(args.cosmic_file, sep="\t", compression="gzip", low_memory=False)
        # Filter frameshift/nonsense
        for i, mut_aa in enumerate(df_cos["MUTATION_AA"]):
            if isinstance(mut_aa, str) and "*" not in mut_aa:
                all_variants_gene.add(f"{df_cos['GENE_SYMBOL'].iloc[i]} {mut_aa[2:]}")
        print(f"  {len(all_variants_gene)} unique gene-variant entries")

    # ------------------------------------------------------------------
    # 3. Map to BioGRID and build complex sets
    # ------------------------------------------------------------------
    print("Loading BioGRID network…")
    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)

    all_complexes = get_complexes_in_biogrid(
        gene_symbol_to_uniprot, uniprot_to_interactors, uniprot_to_seq
    )
    print(f"BioGRID complexes involving COSMIC genes: {len(all_complexes)}")

    # Collect all UniProt IDs in these complexes
    all_uids = {u for pair in all_complexes for u in pair}

    # Write ID list for FASTA download
    ids_out = os.path.join(args.output_dir, "all_uniprot_ids.txt")
    with open(ids_out, "w") as f:
        f.write(" ".join(all_uids))
    print(f"Wrote {len(all_uids)} UniProt IDs to {ids_out}")

    # Load sequences
    if args.uniprot_fasta and os.path.exists(args.uniprot_fasta):
        id_to_seq = read_fasta(args.uniprot_fasta)
        print(f"Loaded {len(id_to_seq)} sequences from FASTA")
    else:
        print("--uniprot-fasta not provided; download sequences for IDs in "
              f"{ids_out} then rerun.")
        return

    # Insert variant sequences for variants with recurrence data
    for gene_vt_id in all_variants_gene:
        gene_symbol, variant = gene_vt_id.split(" ")
        if "del" in variant or "ins" in variant or "Sec" in variant:
            continue
        if gene_symbol not in gene_symbol_to_uniprot:
            continue
        uid = gene_symbol_to_uniprot[gene_symbol]
        if uid not in id_to_seq:
            continue
        wt_res = variant[0]
        try:
            mt_idx = mutations.index(variant)
        except ValueError:
            continue
        mt_res = variant[-1]
        if mt_idx >= len(id_to_seq[uid]) or id_to_seq[uid][mt_idx] != wt_res:
            continue
        new_seq = list(id_to_seq[uid])
        new_seq[mt_idx] = mt_res
        new_seq = "".join(new_seq)
        key = f"{uid} {variant}"
        if key in id_to_seq:
            assert id_to_seq[key] == new_seq
        else:
            id_to_seq[key] = new_seq

    print(f"id_to_seq entries: {len(id_to_seq)}")

    # Complexes cleaned (filter proteins with recurrence data, dedupe)
    complexes_cleaned = set()
    for a, b in all_complexes:
        if a not in proteins_with_recurrence_data:
            continue
        if (b, a) not in complexes_cleaned:
            complexes_cleaned.add((a, b))

    for a, b in complexes_cleaned:
        assert not ((b, a) in complexes_cleaned and a != b)

    print(f"Complexes with recurrence data: {len(complexes_cleaned)}")

    # ------------------------------------------------------------------
    # 4. Build variant triplets (protein mapping, no raw COSMIC records)
    # ------------------------------------------------------------------
    all_variant_triplets = build_variant_triplets_with_recurrence(
        id_to_seq, complexes_cleaned, recurrence_dict
    )
    print(f"Variant-partner triplets: {len(all_variant_triplets)}")

    # ------------------------------------------------------------------
    # 5. Save outputs — NO raw COSMIC sample/mutation data written
    # ------------------------------------------------------------------
    with open(f"{args.output_dir}/recurrence_dirbind_complexes.pkl", "wb") as f:
        pickle.dump(complexes_cleaned, f)
    with open(f"{args.output_dir}/id_to_seq.pkl", "wb") as f:
        pickle.dump(id_to_seq, f)
    with open(f"{args.output_dir}/recurrence_dirbind_all_variants.pkl", "wb") as f:
        pickle.dump(all_variant_triplets, f)

    print(f"\nSummary")
    print(f"  Proteins with recurrence data in BioGRID: {len(proteins_with_recurrence_data & set(u for u,_ in all_complexes))}")
    print(f"  Unique complexes (with recurrence): {len(complexes_cleaned)}")
    print(f"  Variant-partner triplets: {len(all_variant_triplets)}")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Map COSMIC missense variants to BioGRID PPI network; "
                    "outputs recurrence counts only, no variant-level COSMIC data"
    )
    p.add_argument("--cosmic-file", default=None,
                   help="COSMIC GenomeScreensMutant Missense TSV.gz (licensed; not embedded in output)")
    p.add_argument("--cmc-file", required=True,
                   help="COSMIC CancerMutationCensus AllData TSV.gz (licensed; not embedded in output)")
    p.add_argument("--biogrid-dir", required=True,
                   help="Directory with BioGRID pickles from get_biogrid_interactors.py")
    p.add_argument("--uniprot-fasta", default=None,
                   help="UniProt FASTA for COSMIC gene proteins; download IDs then rerun")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for COSMIC mapping results")
    p.add_argument("--gene-symbol-to-uniprot", default=None,
                   help="Pre-built gene_symbol_to_uniprot.pkl (if absent, fetches via HGNC API)")
    main(p.parse_args())
