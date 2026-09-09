#!/usr/bin/env python
"""Map gnomAD missense variants to the BioGRID direct-binding PPI network.

Supports two input modes:
  --mode subset   : reads pre-filtered pkl files (gnomad_wts_sub.pkl,
                    gnomad_vts_sub.pkl) and a RefSeq→UniProt mapping TSV.
  --mode all      : reads a flat text file of 'UNIPROT variant' entries
                    (gnomad_all_validated_missense_variants_uniprot.txt)
                    and a WT sequence FASTA.

Outputs:
  gnomad_uniprot_wts.pkl        : set of UniProt IDs for WT proteins
  gnomad_uniprot_vts.pkl        : set of 'uniprot variant' strings
  all_complexes.pkl             : set of (uniprot_a, uniprot_b) pairs
  complex_subset.pkl            : capped subset (≤10 partners per protein)
  id_to_seq.pkl                 : dict {id: seq} for WTs and VTs
  all_variants.pkl              : set of (uniprot, variant, partner) triplets
  variant_subset.pkl            : triplets corresponding to complex_subset

Usage:
    # Subset mode (small scale)
    python map_gnomad.py --mode subset \
        --gnomad-dir gnomad \
        --biogrid-dir biogrid \
        --output-dir $MUTPRED_DATA_ROOT/gnomad

    # All-variants mode (large scale)
    python map_gnomad.py --mode all \
        --gnomad-variants-txt /path/to/gnomad/gnomad_all_validated_missense_variants_uniprot.txt \
        --wt-fasta /path/to/gnomad/gnomad_all_missense_wild_type_seqs.fasta \
        --biogrid-dir $MUTPRED_DATA_ROOT/biogrid \
        --output-dir $MUTPRED_DATA_ROOT/gnomad \
        --max-complex-subset 2000
"""

import argparse
import os
import pickle
from collections import defaultdict

from Bio import SeqIO
from biogrid_common import (  # shared verbatim helpers
    clean_complexes,
    load_biogrid,
)


# ---------------------------------------------------------------------------
# Helpers shared by both modes
# ---------------------------------------------------------------------------

def get_genes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq):
    """Return set of (uniprot_id, partner) for proteins in both gnomAD and BioGRID."""
    wt_complexes = set()
    for uid in uniprot_wts:
        if uid not in uniprot_to_interactors:
            continue
        for partner in uniprot_to_interactors[uid]:
            if partner in uniprot_to_seq:
                wt_complexes.add((uid, partner))
    return wt_complexes


def sanity_check(all_complexes, id_to_seq, uniprot_to_interactors):
    for a, b in all_complexes:
        assert (
            a in uniprot_to_interactors
            and b in uniprot_to_interactors
            and b in uniprot_to_interactors[a]
            and a in uniprot_to_interactors[b]
        ), f"BioGRID sanity failure: {a}, {b}"
    for vt_id in id_to_seq:
        uid = vt_id.split(" ")[0]
        assert uid in uniprot_to_interactors, f"Protein {uid} not in BioGRID"


def build_complex_subset(all_complexes, max_size):
    """Keep up to 10 partners per mutated protein; stop at max_size total."""
    complex_subset = set()
    for a, b in all_complexes:
        interactor_count = 0
        for na, nb in all_complexes:
            if na == a and interactor_count < 10:
                if (nb, na) not in complex_subset:
                    complex_subset.add((na, nb))
                    interactor_count += 1
                    if len(complex_subset) == max_size:
                        return complex_subset
    return complex_subset


def build_variant_triplets(id_to_seq, complexes):
    """Build (uniprot, variant, partner) triplets from id_to_seq and complex set."""
    # Build a reverse index: uniprot → list of variants
    wt_to_vt = defaultdict(list)
    for key in id_to_seq:
        if " " in key:
            uid, variant = key.split(" ")
            wt_to_vt[uid].append(variant)

    variants = set()
    for uniprot, partner in complexes:
        for v in wt_to_vt.get(uniprot, []):
            variants.add((uniprot, v, partner))
        for v in wt_to_vt.get(partner, []):
            variants.add((partner, v, uniprot))
    return variants


def apply_mutations(uniprot_seq_dict, id_to_seq):
    """Insert variant sequences into id_to_seq for proteins present in id_to_seq."""
    added = 0
    for vt_id, vt_seq in uniprot_seq_dict.items():
        if " " not in vt_id:
            continue
        uid, variant = vt_id.split(" ")
        if "del" in variant or "ins" in variant or "Sec" in variant:
            continue
        wt_res = variant[0]
        try:
            mt_idx = int(variant[1:-1]) - 1
        except ValueError:
            continue
        mt_res = variant[-1]
        if uid not in id_to_seq:
            continue
        if mt_idx >= len(id_to_seq[uid]) or id_to_seq[uid][mt_idx] != wt_res:
            continue
        wt_seq = id_to_seq[uid]
        new_seq = list(wt_seq)
        new_seq[mt_idx] = mt_res
        new_seq = "".join(new_seq)
        key = f"{uid} {variant}"
        if key in id_to_seq:
            assert id_to_seq[key] == new_seq
        else:
            id_to_seq[key] = new_seq
            added += 1
    return added


# ---------------------------------------------------------------------------
# Mode: subset (RefSeq-based, small scale)
# ---------------------------------------------------------------------------

def run_subset_mode(args, uniprot_to_interactors, uniprot_to_seq):
    gnomad_dir = args.gnomad_dir

    with open(f"{gnomad_dir}/gnomad_wts_sub.pkl", "rb") as f:
        all_wts = pickle.load(f)
    with open(f"{gnomad_dir}/gnomad_vts_sub.pkl", "rb") as f:
        all_vts = pickle.load(f)

    # Map RefSeq → UniProt
    df_map = __import__("pandas").read_csv(
        f"{gnomad_dir}/gnomad_refseq_to_uniprot.tsv", sep="\t"
    )
    refseq_to_uniprot = {}
    reviewed = {}
    for _, row in df_map.iterrows():
        rid, uid, rev = row["From"], row["Entry"], row["Reviewed"]
        if rid not in refseq_to_uniprot or (rev == "reviewed" and not reviewed.get(rid, False)):
            refseq_to_uniprot[rid] = uid
            reviewed[rid] = rev == "reviewed"
    # Strip review flag
    refseq_to_uniprot = {k: v for k, v in refseq_to_uniprot.items()}

    # Load WT sequences
    seqs_by_id = defaultdict(list)
    for rec in __import__("Bio.SeqIO", fromlist=["SeqIO"]).parse(
        f"{gnomad_dir}/gnomad_uniprots.fasta", "fasta"
    ):
        acc = rec.id.split("|")[1] if "|" in rec.id else rec.id
        seqs_by_id[acc].append(str(rec.seq))
    uniprot_seq_dict = {}
    for acc, seqs in seqs_by_id.items():
        if len(set(seqs)) > 1:
            raise ValueError(f"Conflicting sequences for {acc}")
        uniprot_seq_dict[acc] = seqs[0]

    # Generate mutant sequences
    for vt_id in all_vts:
        refseq_id, variant = vt_id.split(" ")
        if refseq_id not in refseq_to_uniprot:
            continue
        uid = refseq_to_uniprot[refseq_id]
        if uid not in uniprot_seq_dict:
            continue
        wt_seq = uniprot_seq_dict[uid]
        wt_res, mt_idx, mt_res = variant[0], int(variant[1:-1]) - 1, variant[-1]
        if mt_idx >= len(wt_seq) or wt_seq[mt_idx] != wt_res:
            continue
        vt_seq = list(wt_seq)
        vt_seq[mt_idx] = mt_res
        uniprot_seq_dict[f"{uid} {variant}"] = "".join(vt_seq)

    uniprot_wts = {k for k in uniprot_seq_dict if " " not in k}
    uniprot_vts = {k for k in uniprot_seq_dict if " " in k}
    return uniprot_wts, uniprot_vts, uniprot_seq_dict


# ---------------------------------------------------------------------------
# Mode: all variants (UniProt-native, large scale)
# ---------------------------------------------------------------------------

def run_all_mode(args, uniprot_to_interactors, uniprot_to_seq):
    # Load variant list
    all_vts = set()
    with open(args.gnomad_variants_txt) as f:
        for line in f:
            all_vts.add(line.strip())
    print(f"Loaded {len(all_vts)} gnomAD variants")

    # Load WT sequences
    seqs_by_id = defaultdict(list)
    for rec in SeqIO.parse(args.wt_fasta, "fasta"):
        seqs_by_id[rec.id].append(str(rec.seq))
    uniprot_seq_dict = {}
    for acc, seqs in seqs_by_id.items():
        if len(set(seqs)) > 1:
            raise ValueError(f"Conflicting sequences for {acc}")
        uniprot_seq_dict[acc] = seqs[0]
    print(f"Loaded {len(uniprot_seq_dict)} WT sequences")

    # Generate mutant sequences
    added = 0
    for vt_id in all_vts:
        uid, variant = vt_id.split(" ")
        if uid not in uniprot_seq_dict:
            continue
        wt_seq = uniprot_seq_dict[uid]
        wt_res, mt_idx, mt_res = variant[0], int(variant[1:-1]) - 1, variant[-1]
        if mt_idx >= len(wt_seq) or wt_seq[mt_idx] != wt_res:
            continue
        vt_seq = list(wt_seq)
        vt_seq[mt_idx] = mt_res
        uniprot_seq_dict[f"{uid} {variant}"] = "".join(vt_seq)
        added += 1
    print(f"{added} valid variant sequences generated")

    uniprot_wts = {k for k in uniprot_seq_dict if " " not in k}
    uniprot_vts = {k for k in uniprot_seq_dict if " " in k}
    return uniprot_wts, uniprot_vts, uniprot_seq_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading BioGRID network…")
    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)

    if args.mode == "subset":
        uniprot_wts, uniprot_vts, uniprot_seq_dict = run_subset_mode(
            args, uniprot_to_interactors, uniprot_to_seq
        )
    else:
        uniprot_wts, uniprot_vts, uniprot_seq_dict = run_all_mode(
            args, uniprot_to_interactors, uniprot_to_seq
        )

    print(f"WT proteins: {len(uniprot_wts)}, VT entries: {len(uniprot_vts)}")

    # Save WT/VT ID sets
    with open(f"{args.output_dir}/gnomad_uniprot_wts.pkl", "wb") as f:
        pickle.dump(uniprot_wts, f)
    with open(f"{args.output_dir}/gnomad_uniprot_vts.pkl", "wb") as f:
        pickle.dump(uniprot_vts, f)

    # Find complexes in BioGRID
    all_complexes = get_genes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq)
    print(f"BioGRID complexes involving gnomAD proteins: {len(all_complexes)}")

    # Collect all BioGRID protein IDs in these complexes
    all_uids_in_biogrid = set()
    for a, b in all_complexes:
        all_uids_in_biogrid.add(a)
        all_uids_in_biogrid.add(b)

    # Build id_to_seq: WT from BioGRID seqs, VT from gnomAD
    id_to_seq = {uid: uniprot_to_seq[uid]
                 for uid in all_uids_in_biogrid
                 if uid in uniprot_to_seq}

    added = apply_mutations(uniprot_seq_dict, id_to_seq)
    print(f"Added {added} mutant sequences to id_to_seq; total: {len(id_to_seq)}")

    # Sanity check
    sanity_check(all_complexes, id_to_seq, uniprot_to_interactors)

    # Clean and subset complexes
    complexes_cleaned = clean_complexes(all_complexes)
    complex_subset = build_complex_subset(all_complexes, args.max_complex_subset)
    print(f"Cleaned complexes: {len(complexes_cleaned)}, subset: {len(complex_subset)}")

    # Verify no flipped duplicates in cleaned set
    for a, b in complexes_cleaned:
        assert not ((b, a) in complexes_cleaned and a != b)

    # Build variant triplets
    all_variants = build_variant_triplets(id_to_seq, complexes_cleaned)
    variant_subset = build_variant_triplets(id_to_seq, complex_subset)
    print(f"Variant triplets — all: {len(all_variants)}, subset: {len(variant_subset)}")

    # Save outputs
    with open(f"{args.output_dir}/all_complexes.pkl", "wb") as f:
        pickle.dump(complexes_cleaned, f)
    with open(f"{args.output_dir}/complex_subset.pkl", "wb") as f:
        pickle.dump(complex_subset, f)
    with open(f"{args.output_dir}/id_to_seq.pkl", "wb") as f:
        pickle.dump(id_to_seq, f)
    with open(f"{args.output_dir}/all_variants.pkl", "wb") as f:
        pickle.dump(all_variants, f)
    with open(f"{args.output_dir}/variant_subset.pkl", "wb") as f:
        pickle.dump(variant_subset, f)

    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Map gnomAD missense variants to BioGRID PPI network"
    )
    p.add_argument("--mode", choices=["subset", "all"], default="all",
                   help="'subset': RefSeq pkl input; 'all': full UniProt text input")
    # subset mode inputs
    p.add_argument("--gnomad-dir", default=None,
                   help="Directory with gnomad_wts_sub.pkl, gnomad_vts_sub.pkl, etc.")
    # all mode inputs
    p.add_argument("--gnomad-variants-txt", default=None,
                   help="Text file with one 'UNIPROT variant' per line (all-mode)")
    p.add_argument("--wt-fasta", default=None,
                   help="FASTA of gnomAD WT protein sequences (all-mode)")
    # shared
    p.add_argument("--biogrid-dir", required=True,
                   help="Directory with BioGRID pickles from get_biogrid_interactors.py")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for gnomAD pickles")
    p.add_argument("--max-complex-subset", type=int, default=2000,
                   help="Maximum number of complexes in the capped subset (default: 2000)")
    main(p.parse_args())
