#!/usr/bin/env python
"""Prepare the Sahni et al. Y2H training dataset for MutPred-PPI.

Reads the Sahni variant CSV (with Edgotype_class labels), fetches protein
sequences from NCBI if needed, maps RefSeq IDs to UniProt, generates mutated
sequences, and writes a training CSV with columns
(refseq_id, Mutation, partner, Y2H_score).

Usage:
    python prepare_sahni.py \
        --variants-csv variants.csv \
        --refseq-to-uniprot refseq_to_uniprot.tsv \
        --biogrid-interactors biogrid/biogrid_dirbind_uniprot_to_interactors.pkl \
        --biogrid-seqs biogrid/uniprot_dirbind_to_seq.pkl \
        --output-dir $MUTPRED_DATA_ROOT/home/sahni \
        --email your@email.com
"""

import argparse
import os
import pickle
import pandas as pd
from Bio import Entrez, SeqIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_protein_sequences(refseq_ids, email):
    """Fetch protein sequences from NCBI for a list of RefSeq IDs."""
    Entrez.email = email
    ids_string = ",".join(refseq_ids)
    handle = Entrez.efetch(db="protein", id=ids_string, rettype="fasta", retmode="text")
    records = list(SeqIO.parse(handle, "fasta"))
    handle.close()
    return {rec.id.split(".")[0]: str(rec.seq) for rec in records}


def load_refseq_to_uniprot(tsv_path):
    """Parse a UniProt ID-mapping TSV (From → Entry → Reviewed columns)."""
    df = pd.read_csv(tsv_path, sep="\t")
    mapping = {}
    reviewed = {}
    for _, row in df.iterrows():
        refseq_id = row["From"]
        uniprot_id = row["Entry"]
        is_reviewed = row.get("Reviewed", "") == "reviewed"
        if refseq_id not in mapping or (is_reviewed and not reviewed.get(refseq_id, False)):
            mapping[refseq_id] = uniprot_id
            reviewed[refseq_id] = is_reviewed
    return mapping


def make_variant_key(refseq_id, variant):
    return f"{refseq_id} {variant}"


def generate_mutated_seq(wt_seq, variant):
    """Apply a single amino-acid substitution and return the mutant sequence.

    variant format: <wt_aa><1-based_position><mt_aa>, e.g. 'A42V'.
    Returns None if the WT residue does not match.
    """
    wt_res = variant[0]
    try:
        pos = int(variant[1:-1]) - 1
    except ValueError:
        return None
    mt_res = variant[-1]
    if pos >= len(wt_seq) or wt_seq[pos] != wt_res:
        return None
    vt_seq = list(wt_seq)
    vt_seq[pos] = mt_res
    return "".join(vt_seq)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Sahni variant table
    # ------------------------------------------------------------------
    df = pd.read_csv(args.variants_csv)
    print(f"Loaded {len(df)} variants from {args.variants_csv}")

    # Build Loss / Gain labels from Edgotype_class
    # (Y2H_score convention: 1 = interaction disrupted, 0 = interaction retained)
    loss_l, gain_l = [], []
    for _, row in df.iterrows():
        edgotype = row["Edgotype_class"]
        loss = 1 if edgotype in ["Edgetic", "Gain-and-loss-of-interaction", "Quasi-null"] else 0
        gain = 1 if edgotype in ["Gain-of-interaction", "Gain-and-loss-of-interaction"] else 0
        loss_l.append(loss)
        gain_l.append(gain)
    df["Loss"] = loss_l
    df["Gain"] = gain_l

    # Extract RefSeq IDs and variant strings from Mutation_RefSeq_AA column
    # Expected format: NP_XXXXXX:p.A42V
    df["refseq_id"] = df["Mutation_RefSeq_AA"].str.split(":p.").str[0]
    df["Mutation"] = df["Mutation_RefSeq_AA"].str.split(":p.").str[1]

    refseq_ids = df["refseq_id"].unique().tolist()
    print(f"Unique RefSeq IDs: {len(refseq_ids)}")

    # ------------------------------------------------------------------
    # 2. Load or fetch protein sequences
    # ------------------------------------------------------------------
    if args.sequences_fasta and os.path.exists(args.sequences_fasta):
        id_to_seq = {}
        for rec in SeqIO.parse(args.sequences_fasta, "fasta"):
            id_to_seq[rec.id.split(".")[0]] = str(rec.seq)
        print(f"Loaded {len(id_to_seq)} sequences from {args.sequences_fasta}")
    else:
        print("Fetching sequences from NCBI (this may take a while)…")
        id_to_seq = fetch_protein_sequences(refseq_ids, args.email)
        # Cache for reuse
        fasta_out = os.path.join(args.output_dir, "refseq_wts.fasta")
        with open(fasta_out, "w") as f:
            for id_, seq in id_to_seq.items():
                f.write(f">{id_}\n{seq}\n")
        print(f"Saved {len(id_to_seq)} sequences to {fasta_out}")

    # ------------------------------------------------------------------
    # 3. Map RefSeq → UniProt
    # ------------------------------------------------------------------
    refseq_to_uniprot = load_refseq_to_uniprot(args.refseq_to_uniprot)
    print(f"Loaded {len(refseq_to_uniprot)} RefSeq→UniProt mappings")

    uniprot_to_seq = {}
    for refseq_id, uniprot_id in refseq_to_uniprot.items():
        if refseq_id in id_to_seq:
            uniprot_to_seq[uniprot_id] = id_to_seq[refseq_id]

    # Save RefSeq→UniProt pickle for downstream use
    pkl_out = os.path.join(args.output_dir, "refseq_to_uniprot.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(refseq_to_uniprot, f)
    print(f"Saved RefSeq→UniProt mapping to {pkl_out}")

    # ------------------------------------------------------------------
    # 4. Build id_to_seq with mutated sequences
    # ------------------------------------------------------------------
    id_to_seq_uniprot = dict(uniprot_to_seq)
    bad_rows = []
    for i, row in df.iterrows():
        refseq_id = row["refseq_id"]
        variant = row["Mutation"]
        if refseq_id not in refseq_to_uniprot:
            bad_rows.append(i)
            continue
        uniprot_id = refseq_to_uniprot[refseq_id]
        if uniprot_id not in id_to_seq_uniprot:
            bad_rows.append(i)
            continue
        wt_seq = id_to_seq_uniprot[uniprot_id]
        vt_seq = generate_mutated_seq(wt_seq, variant)
        if vt_seq is None:
            bad_rows.append(i)
            continue
        id_to_seq_uniprot[f"{uniprot_id} {variant}"] = vt_seq

    df_clean = df.drop(bad_rows).reset_index(drop=True)
    print(f"Retained {len(df_clean)} / {len(df)} variants after sequence validation")

    # ------------------------------------------------------------------
    # 5. Map to BioGRID interactors and build training triplets
    # ------------------------------------------------------------------
    with open(args.biogrid_interactors, "rb") as f:
        uniprot_to_interactors = pickle.load(f)
    with open(args.biogrid_seqs, "rb") as f:
        biogrid_uniprot_to_seq = pickle.load(f)

    rows_out = []
    for _, row in df_clean.iterrows():
        refseq_id = row["refseq_id"]
        variant = row["Mutation"]
        if refseq_id not in refseq_to_uniprot:
            continue
        uniprot_id = refseq_to_uniprot[refseq_id]
        if uniprot_id not in uniprot_to_interactors:
            continue
        y2h = 1 if row["Loss"] == 1 else 0
        for partner in uniprot_to_interactors[uniprot_id]:
            if partner in biogrid_uniprot_to_seq:
                rows_out.append({
                    "refseq_id": uniprot_id,
                    "Mutation": variant,
                    "partner": partner,
                    "Y2H_score": y2h,
                })

    df_out = pd.DataFrame(rows_out)
    out_csv = os.path.join(args.output_dir, "sahni_train.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_out)} training rows to {out_csv}")

    # ------------------------------------------------------------------
    # 6. Write FASTA for structure prediction
    # ------------------------------------------------------------------
    combined_id_to_seq = {}
    for uniprot_id in uniprot_to_interactors:
        if uniprot_id in biogrid_uniprot_to_seq:
            combined_id_to_seq[uniprot_id] = biogrid_uniprot_to_seq[uniprot_id]
    combined_id_to_seq.update(id_to_seq_uniprot)

    fasta_out = os.path.join(args.output_dir, "sahni_wts_and_vts.fasta")
    with open(fasta_out, "w") as f:
        for id_, seq in combined_id_to_seq.items():
            f.write(f">{id_}\n{seq}\n")
    print(f"Wrote combined WT/VT FASTA to {fasta_out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Prepare Sahni Y2H training dataset for MutPred-PPI")
    p.add_argument("--variants-csv", required=True,
                   help="Sahni variant CSV with Edgotype_class and Mutation_RefSeq_AA columns")
    p.add_argument("--refseq-to-uniprot", required=True,
                   help="UniProt ID mapping TSV (From/Entry/Reviewed columns)")
    p.add_argument("--biogrid-interactors", required=True,
                   help="Pickle of BioGRID direct-binding uniprot_to_interactors dict")
    p.add_argument("--biogrid-seqs", required=True,
                   help="Pickle of BioGRID UniProt→sequence dict")
    p.add_argument("--output-dir", required=True,
                   help="Directory for output files (sahni_train.csv, FASTA, pickles)")
    p.add_argument("--sequences-fasta", default=None,
                   help="Pre-fetched WT sequences FASTA; if absent, fetches from NCBI")
    p.add_argument("--email", default="",
                   help="Email for NCBI Entrez (required when fetching sequences)")
    main(p.parse_args())
