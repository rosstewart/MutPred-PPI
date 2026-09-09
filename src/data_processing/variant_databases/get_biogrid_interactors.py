#!/usr/bin/env python
"""Extract direct-binding PPI partners from BioGRID and map ClinVar variants to them.

Stage 1 (BioGRID): reads the BioGRID full interaction TSV, keeps only
physically binding experimental systems (Co-crystal Structure, Far Western,
Reconstituted Complex, Cross-Linking-MS), and writes a uniprot_to_interactors
pickle and a FASTA of interactor sequences.

Stage 2 (ClinVar): reads ClinVar WT+VT FASTA files (one per significance
class), intersects with the BioGRID network, and writes per-class variant
subsets as pickles.

Usage:
    # Stage 1 only (BioGRID)
    python get_biogrid_interactors.py \
        --biogrid-tsv biogrid/biogrid_ppi.tsv \
        --uniprot-fasta biogrid/all_uniprot_ids.fasta \
        --output-dir $MUTPRED_DATA_ROOT/biogrid

    # Stage 1 + Stage 2 (ClinVar)
    python get_biogrid_interactors.py \
        --biogrid-tsv biogrid/biogrid_ppi.tsv \
        --uniprot-fasta biogrid/all_uniprot_ids.fasta \
        --output-dir $MUTPRED_DATA_ROOT/biogrid \
        --clinvar-dir /path/to/clinvar \
        --clinvar-output-dir $MUTPRED_DATA_ROOT/clinvar \
        --train-uniprot-ids combined_train_uniprot_ids.pkl
"""

import argparse
import os
import pickle
import pandas as pd


# ---------------------------------------------------------------------------
# Binding-evidence experimental systems
# ---------------------------------------------------------------------------

BINDING_TECHNIQUES = {
    "Co-crystal Structure",
    "Cross-Linking-MS (XL-MS)",
    "Far Western",
    "Reconstituted Complex",
    "Protein-Peptide",
}


# ---------------------------------------------------------------------------
# BioGRID helpers
# ---------------------------------------------------------------------------

def build_uniprot_to_interactors(df):
    """Return dict {uniprot_id: [partner_uniprot_ids]} for direct-binding PPIs."""
    uniprot_to_interactors = {}
    for _, row in df.iterrows():
        if row["Experimental System Type"] != "physical":
            continue
        if row["Experimental System"] not in BINDING_TECHNIQUES:
            continue
        id_a = row["SWISS-PROT Accessions Interactor A"]
        id_b = row["SWISS-PROT Accessions Interactor B"]
        for x, y in [(id_a, id_b), (id_b, id_a)]:
            if x not in uniprot_to_interactors:
                uniprot_to_interactors[x] = []
            if y not in uniprot_to_interactors[x]:
                uniprot_to_interactors[x].append(y)
    return uniprot_to_interactors


def collect_all_uniprot_ids(uniprot_to_interactors):
    ids = set()
    for protein, partners in uniprot_to_interactors.items():
        ids.add(protein)
        ids.update(partners)
    return ids


def load_uniprot_seqs_from_fasta(fasta_path):
    """Load UniProt sequences from a standard UniProt FASTA (>sp|ACC|NAME …)."""
    from Bio import SeqIO
    seq_dict = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        try:
            acc = rec.id.split("|")[1]
        except IndexError:
            acc = rec.id
        seq_dict[acc] = str(rec.seq)
    return seq_dict


# ---------------------------------------------------------------------------
# ClinVar helpers
# ---------------------------------------------------------------------------

def get_all_vts_from_fasta(fasta_path):
    """Return set of 'uniprot_id variant' strings parsed from ClinVar FASTA headers."""
    all_vts = set()
    with open(fasta_path) as f:
        for line in f:
            if line[0] != ">":
                continue
            parts = line[1:].strip().split(" ")
            if len(parts) == 2:
                all_vts.add(f"{parts[0]} {parts[1]}")
    return all_vts


def build_clinvar_id_to_seq(fasta_path, uniprot_to_interactors, uniprot_to_seq):
    """
    Parse a ClinVar FASTA and, for each variant on a protein in the BioGRID
    network, store the WT and VT sequences plus partner sequences.

    Returns:
        id_to_seq  : dict {uniprot_id: seq, 'uniprot_id variant': seq, ...}
        wt_complexes : set of (uniprot_id, partner) tuples
    """
    id_to_seq = {}
    wt_complexes = set()

    current_id = None
    current_variant = None
    current_seq = None

    def _store():
        nonlocal current_id, current_variant, current_seq
        if current_id is None or current_seq is None:
            return
        key = f"{current_id} {current_variant}" if current_variant else current_id
        id_to_seq[key] = current_seq

        if current_id in uniprot_to_interactors:
            for partner in uniprot_to_interactors[current_id]:
                if partner not in id_to_seq and partner in uniprot_to_seq:
                    id_to_seq[partner] = uniprot_to_seq[partner]
                if current_variant is None and partner in uniprot_to_seq:
                    wt_complexes.add((current_id, partner))

    with open(fasta_path) as f:
        for line in f:
            if line[0] == ">":
                _store()
                header = line[1:].strip()
                parts = header.split(" ")
                if len(parts) == 2:
                    current_id, current_variant = parts
                    if len(parts[1].split(" ")) != 1:
                        current_id = current_variant = None
                        current_seq = None
                        continue
                elif len(parts) == 1:
                    current_id = parts[0]
                    current_variant = None
                else:
                    current_id = current_variant = None
                    current_seq = None
                    continue

                if current_id not in uniprot_to_interactors:
                    current_id = None
                current_seq = None
            else:
                if current_id is not None:
                    if current_seq is None:
                        current_seq = line.strip()
                    else:
                        current_seq += line.strip()
        _store()

    n_variants = sum(1 for k in id_to_seq if " " in k)
    print(f"  {len(id_to_seq)} sequences ({n_variants} variants)")
    return id_to_seq, wt_complexes


def remove_cross_class_variants(dicts):
    """Remove variant IDs that appear in more than one significance class."""
    variant_keys = [
        {k for k in d if " " in k}
        for d in dicts
    ]
    shared = set()
    for i in range(len(variant_keys)):
        for j in range(i + 1, len(variant_keys)):
            shared |= variant_keys[i] & variant_keys[j]
    for d in dicts:
        for k in shared:
            d.pop(k, None)
    return shared


def build_variant_subset(id_to_seq, complexes_cleaned):
    """Enumerate (uniprot_id, variant, partner) triplets."""
    variant_subset = set()
    for key in id_to_seq:
        if " " not in key:
            continue
        uniprot_id, variant = key.split(" ")
        for uniprot, partner in complexes_cleaned:
            if uniprot_id == uniprot:
                variant_subset.add((uniprot_id, variant, partner))
            if uniprot_id == partner:
                variant_subset.add((uniprot_id, variant, uniprot))
    return variant_subset


def clean_complexes(complexes):
    """Remove flipped duplicates, keeping one canonical orientation."""
    cleaned = set()
    for a, b in complexes:
        if (b, a) not in cleaned:
            cleaned.add((a, b))
    return cleaned


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    # ======================================================================
    # Stage 1: BioGRID
    # ======================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading BioGRID TSV…")
    df_bg = pd.read_csv(args.biogrid_tsv, sep="\t", low_memory=False)
    print(f"  {len(df_bg)} interactions loaded")

    uniprot_to_interactors = build_uniprot_to_interactors(df_bg)
    print(f"  {len(uniprot_to_interactors)} proteins with direct-binding partners")

    # Save interactor dict
    pkl_out = os.path.join(args.output_dir, "biogrid_dirbind_uniprot_to_interactors.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(uniprot_to_interactors, f)
    print(f"Saved interactor dict to {pkl_out}")

    # Write all uniprot IDs for sequence retrieval
    all_ids = collect_all_uniprot_ids(uniprot_to_interactors)
    ids_out = os.path.join(args.output_dir, "biogrid_all_uniprot_ids.txt")
    with open(ids_out, "w") as f:
        for uid in all_ids:
            if "|" in uid:
                for sub_id in uid.split("|"):
                    f.write(f"{sub_id} ")
            else:
                f.write(f"{uid} ")
    print(f"Wrote {len(all_ids)} IDs to {ids_out}")

    # Load sequences (requires pre-downloaded FASTA; see IDs file above)
    if args.uniprot_fasta and os.path.exists(args.uniprot_fasta):
        print("Loading UniProt sequences from FASTA…")
        uniprot_to_seq = load_uniprot_seqs_from_fasta(args.uniprot_fasta)
        print(f"  {len(uniprot_to_seq)} sequences loaded")

        seq_pkl_out = os.path.join(args.output_dir, "uniprot_dirbind_to_seq.pkl")
        with open(seq_pkl_out, "wb") as f:
            pickle.dump(uniprot_to_seq, f)
        print(f"Saved sequence dict to {seq_pkl_out}")

        # Write cleaned FASTA with only BioGRID sequences
        fasta_out = os.path.join(args.output_dir, "biogrid_dirbind_uniprot_ids_clean.fasta")
        with open(fasta_out, "w") as f:
            for uid, seq in uniprot_to_seq.items():
                if uid in all_ids:
                    f.write(f">{uid}\n{seq}\n")
        print(f"Saved cleaned FASTA to {fasta_out}")
    else:
        print(
            "No --uniprot-fasta provided or file not found; "
            "download sequences for IDs in biogrid_all_uniprot_ids.txt and rerun."
        )
        if not args.clinvar_dir:
            return
        uniprot_to_seq = {}
        seq_pkl_out = os.path.join(args.output_dir, "uniprot_dirbind_to_seq.pkl")
        if os.path.exists(seq_pkl_out):
            with open(seq_pkl_out, "rb") as f:
                uniprot_to_seq = pickle.load(f)

    if not args.clinvar_dir:
        return

    # ======================================================================
    # Stage 2: ClinVar
    # ======================================================================
    os.makedirs(args.clinvar_output_dir, exist_ok=True)

    # Load training protein IDs to exclude
    train_uniprot_ids = set()
    if args.train_uniprot_ids and os.path.exists(args.train_uniprot_ids):
        with open(args.train_uniprot_ids, "rb") as f:
            train_uniprot_ids = pickle.load(f)
        print(f"Loaded {len(train_uniprot_ids)} training UniProt IDs to exclude")

    significance_classes = ["benign", "pathogenic", "vus"]
    all_id_to_seq_dicts = []
    all_complexes_per_class = []

    for sig in significance_classes:
        fasta_path = os.path.join(args.clinvar_dir, f"{sig}_wts_and_mts_04_25.fasta")
        if not os.path.exists(fasta_path):
            print(f"ClinVar FASTA not found: {fasta_path} — skipping {sig}")
            all_id_to_seq_dicts.append({})
            all_complexes_per_class.append(set())
            continue

        print(f"Processing {sig}…")
        id_to_seq, wt_complexes = build_clinvar_id_to_seq(
            fasta_path, uniprot_to_interactors, uniprot_to_seq
        )
        all_id_to_seq_dicts.append(id_to_seq)
        all_complexes_per_class.append(wt_complexes)

        # Save all-variants set (protein+variant only, no clinical data)
        all_vts = get_all_vts_from_fasta(fasta_path)
        vts_pkl = os.path.join(args.clinvar_output_dir, f"clinvar_{sig}_uniprot_vts.pkl")
        with open(vts_pkl, "wb") as f:
            pickle.dump(all_vts, f)

        fasta_combined = os.path.join(
            args.clinvar_output_dir, f"clinvar_{sig}_wt_vt_partners.fasta"
        )
        with open(fasta_combined, "w") as f:
            for id_, seq in id_to_seq.items():
                f.write(f">{id_}\n{seq}\n")
        print(f"  Wrote {fasta_combined}")

    # Remove variants present in >1 class
    shared = remove_cross_class_variants(all_id_to_seq_dicts)
    if shared:
        print(f"Removed {len(shared)} cross-class variants")

    # Build all_complexes (union, deduped)
    all_complexes_union = set()
    for c in all_complexes_per_class:
        all_complexes_union |= c
    non_duplicate_complexes = clean_complexes(all_complexes_union)
    print(f"Total unique complexes: {len(non_duplicate_complexes)}")

    # Build per-class variant subsets (triplets)
    for sig, id_to_seq in zip(significance_classes, all_id_to_seq_dicts):
        variant_subset = build_variant_subset(id_to_seq, non_duplicate_complexes)
        pkl_out = os.path.join(args.clinvar_output_dir, f"{sig}_dirbind_variant_subset.pkl")
        with open(pkl_out, "wb") as f:
            pickle.dump(variant_subset, f)
        print(f"  {sig}: {len(variant_subset)} (protein, variant, partner) triplets → {pkl_out}")

    # Save combined id_to_seq and complex set
    all_id_to_seq = {}
    for d in all_id_to_seq_dicts:
        all_id_to_seq.update(d)
    combined_pkl = os.path.join(args.clinvar_output_dir, "dirbind_with_vus_id_to_seq.pkl")
    with open(combined_pkl, "wb") as f:
        pickle.dump(all_id_to_seq, f)

    complexes_pkl = os.path.join(args.clinvar_output_dir, "non_duplicate_complexes.pkl")
    with open(complexes_pkl, "wb") as f:
        pickle.dump(non_duplicate_complexes, f)
    print(f"Saved combined id_to_seq to {combined_pkl}")
    print(f"Saved complex set to {complexes_pkl}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build BioGRID direct-binding interactor network and map ClinVar variants"
    )
    p.add_argument("--biogrid-tsv", required=True,
                   help="BioGRID full interaction TSV (BIOGRID-ALL-*.tab3.txt)")
    p.add_argument("--uniprot-fasta", default=None,
                   help="UniProt FASTA for BioGRID proteins; download IDs first then rerun")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for BioGRID pickles and FASTA")
    # ClinVar stage (optional)
    p.add_argument("--clinvar-dir", default=None,
                   help="Directory containing ClinVar FASTA files (benign/pathogenic/vus)")
    p.add_argument("--clinvar-output-dir", default=None,
                   help="Output directory for ClinVar variant subsets")
    p.add_argument("--train-uniprot-ids", default=None,
                   help="Pickle of training-set UniProt IDs to exclude")
    main(p.parse_args())
