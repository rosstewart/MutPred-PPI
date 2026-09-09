#!/usr/bin/env python
"""Map autism/NDD variant datasets to the BioGRID direct-binding PPI network.

Handles two complementary datasets:

  NeuroDev (--mode neurodev):
    Reads neurodev_case.fasta and neurodev_control.fasta, maps RefSeq IDs to
    UniProt via a TSV, validates mutated sequences, and writes (uniprot,
    variant, partner) triplets with case/control labels.

  Fu/Tulika autism (--mode tulika):
    Reads Fu_variants_SebatLab.tsv and Fu_variants_LIT.tsv, fetches UniProt
    IDs from HGNC REST, and writes all-complex variant triplets.

  Combined (--mode both, default):
    Runs both stages in sequence.

Outputs (per mode):
  id_to_seq.pkl            : WT + VT sequences
  all_complexes.pkl        : (uniprot_a, uniprot_b) complex pairs
  all_variants.pkl         : (uniprot, variant, partner) triplets
  variant_subset.pkl       : capped complex subset triplets (NeuroDev only)
  variant_label_dict.pkl   : {uniprot variant: 0/1} case/control labels (NeuroDev)

Usage:
    # NeuroDev only
    python map_autism.py --mode neurodev \
        --neurodev-case autism/neurodev_case.fasta \
        --neurodev-control autism/neurodev_control.fasta \
        --neurodev-refseq-to-uniprot autism/neurodev_refseq_to_uniprot.tsv \
        --biogrid-dir biogrid \
        --output-dir $MUTPRED_DATA_ROOT/autism \
        --max-complex-subset 720

    # Fu/Tulika autism only
    python map_autism.py --mode tulika \
        --tulika-dir tulika_autism \
        --tulika-fasta tulika_autism/tulika_autism_uniprots.fasta \
        --biogrid-dir biogrid \
        --output-dir $MUTPRED_DATA_ROOT/tulika_autism

    # Both
    python map_autism.py --mode both \
        --neurodev-case autism/neurodev_case.fasta \
        --neurodev-control autism/neurodev_control.fasta \
        --neurodev-refseq-to-uniprot autism/neurodev_refseq_to_uniprot.tsv \
        --tulika-dir tulika_autism \
        --tulika-fasta tulika_autism/tulika_autism_uniprots.fasta \
        --biogrid-dir biogrid \
        --neurodev-output-dir $MUTPRED_DATA_ROOT/autism \
        --tulika-output-dir $MUTPRED_DATA_ROOT/tulika_autism
"""

import argparse
import os
import pickle
import pandas as pd
import requests
from Bio import SeqIO
from biogrid_common import (  # shared verbatim helpers
    build_variant_triplets,
    clean_complexes,
    get_complexes_in_biogrid,
    load_biogrid,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_complex_subset(all_complexes, max_size):
    complex_subset = set()
    for a, b in all_complexes:
        count = 0
        for na, nb in all_complexes:
            if na == a and count < 10:
                if (nb, na) not in complex_subset:
                    complex_subset.add((na, nb))
                    count += 1
                    if len(complex_subset) == max_size:
                        return complex_subset
    return complex_subset


def apply_variants(variant_seq_dict, id_to_seq):
    """Insert VT sequences into id_to_seq for proteins already present."""
    for vt_id, vt_seq in variant_seq_dict.items():
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
        new_seq = list(id_to_seq[uid])
        new_seq[mt_idx] = mt_res
        new_seq = "".join(new_seq)
        key = f"{uid} {variant}"
        if key in id_to_seq:
            assert id_to_seq[key] == new_seq
        else:
            id_to_seq[key] = new_seq


def sanity_check(all_complexes, id_to_seq, uniprot_to_interactors):
    for a, b in all_complexes:
        assert (a in uniprot_to_interactors and b in uniprot_to_interactors
                and b in uniprot_to_interactors[a] and a in uniprot_to_interactors[b])
    for vt_id in id_to_seq:
        uid = vt_id.split(" ")[0]
        assert uid in uniprot_to_interactors


def load_refseq_to_uniprot(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    mapping, reviewed = {}, {}
    for _, row in df.iterrows():
        rid, uid, rev = row["From"], row["Entry"], row["Reviewed"]
        is_rev = rev == "reviewed"
        if rid not in mapping or (is_rev and not reviewed.get(rid, False)):
            mapping[rid] = uid
            reviewed[rid] = is_rev
    return mapping


# ---------------------------------------------------------------------------
# NeuroDev pipeline
# ---------------------------------------------------------------------------

def merge_dicts_no_conflict(*dicts):
    result = {}
    for d in dicts:
        conflicting = result.keys() & d.keys()
        assert not conflicting, f"Conflicting keys: {conflicting}"
        result.update(d)
    return result


def run_neurodev(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("=== NeuroDev mode ===")

    # Load case and control FASTA
    seq_dict_case = {str(r.id): str(r.seq)
                     for r in SeqIO.parse(args.neurodev_case, "fasta")}
    seq_dict_control = {str(r.id): str(r.seq)
                        for r in SeqIO.parse(args.neurodev_control, "fasta")}

    # Remove shared variants
    shared = [k for k in seq_dict_control if k in seq_dict_case
              and seq_dict_control[k] == seq_dict_case[k]]
    for k in shared:
        del seq_dict_control[k]
        del seq_dict_case[k]

    seq_dict_full = merge_dicts_no_conflict(seq_dict_control, seq_dict_case)
    print(f"  Combined case+control: {len(seq_dict_full)} entries")

    # RefSeq → UniProt
    refseq_ids = set(k.split("|")[0] for k in seq_dict_full)
    refseq_to_uniprot = load_refseq_to_uniprot(args.neurodev_refseq_to_uniprot)
    print(f"  {len(refseq_to_uniprot)} RefSeq→UniProt mappings")

    # Write IDs for downstream UniProt lookup
    ids_out = os.path.join(output_dir, "neurodev_refseq_ids.txt")
    with open(ids_out, "w") as f:
        f.write(" ".join(refseq_ids))

    # Build uniprot_seq_dict with WT seqs (un-mutate the variant FASTA entries)
    uniprot_seq_dict = {}
    variant_label_dict = {}
    uniprots_to_remove = set()

    for key, variant_seq in seq_dict_full.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        refseq_id, gene, variant = parts
        variant = variant[2:]  # strip 'p.'
        wt_res = variant[0]
        try:
            mt_idx = int(variant[1:-1]) - 1
        except ValueError:
            continue
        mt_res = variant[-1]

        if refseq_id not in refseq_to_uniprot:
            continue
        uid = refseq_to_uniprot[refseq_id]
        new_key = f"{uid} {variant}"

        label = 1 if key in seq_dict_case else 0
        assert variant_seq[mt_idx] == mt_res

        # Un-mutate to get WT
        wt_seq = list(variant_seq)
        wt_seq[mt_idx] = wt_res
        wt_seq = "".join(wt_seq)

        if uid not in uniprot_seq_dict:
            uniprot_seq_dict[uid] = wt_seq
        elif uniprot_seq_dict[uid] != wt_seq:
            uniprots_to_remove.add(uid)

        assert new_key not in variant_label_dict
        variant_label_dict[new_key] = label
        uniprot_seq_dict[new_key] = variant_seq

    # Remove proteins with conflicting WT sequences
    keys_to_remove = {k for k in uniprot_seq_dict
                      if k.split(" ")[0] in uniprots_to_remove}
    for k in keys_to_remove:
        del uniprot_seq_dict[k]
    print(f"  After removing conflicting WT seqs: {len(uniprot_seq_dict)} entries")

    uniprot_wts = {k for k in uniprot_seq_dict if " " not in k}

    # BioGRID intersection
    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)
    all_complexes = get_complexes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq)
    print(f"  BioGRID complexes: {len(all_complexes)}")

    all_uids = {u for pair in all_complexes for u in pair}
    id_to_seq = {uid: uniprot_to_seq[uid] for uid in all_uids if uid in uniprot_to_seq}
    apply_variants(uniprot_seq_dict, id_to_seq)

    sanity_check(all_complexes, id_to_seq, uniprot_to_interactors)

    complexes_cleaned = clean_complexes(all_complexes)
    complex_subset = build_complex_subset(all_complexes, args.max_complex_subset)
    for a, b in complexes_cleaned:
        assert not ((b, a) in complexes_cleaned and a != b)

    all_variants = build_variant_triplets(id_to_seq, complexes_cleaned)
    variant_subset = build_variant_triplets(id_to_seq, complex_subset)
    print(f"  Variant triplets — all: {len(all_variants)}, subset: {len(variant_subset)}")

    # Save
    with open(f"{output_dir}/all_complexes.pkl", "wb") as f:
        pickle.dump(complexes_cleaned, f)
    with open(f"{output_dir}/complex_subset.pkl", "wb") as f:
        pickle.dump(complex_subset, f)
    with open(f"{output_dir}/id_to_seq.pkl", "wb") as f:
        pickle.dump(id_to_seq, f)
    with open(f"{output_dir}/all_variants.pkl", "wb") as f:
        pickle.dump(all_variants, f)
    with open(f"{output_dir}/variant_subset.pkl", "wb") as f:
        pickle.dump(variant_subset, f)
    with open(f"{output_dir}/variant_label_dict.pkl", "wb") as f:
        pickle.dump(variant_label_dict, f)
    print(f"  Outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Tulika/Fu autism pipeline
# ---------------------------------------------------------------------------

def fetch_hgnc_info(gene_list):
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


def build_gene_to_uniprot(df_hgnc):
    mapping = {}
    for _, row in df_hgnc.iterrows():
        gene = row["Gene Symbol"]
        uid = row["UniProt ID"]
        if ", " in uid:
            # Use first; prefer entries with an AF4 structure if available
            uid = uid.split(", ")[0]
        if gene not in mapping:
            mapping[gene] = uid
    return mapping


def run_tulika(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("=== Tulika/Fu autism mode ===")

    # Load variant TSV files
    df1 = pd.read_csv(os.path.join(args.tulika_dir, "Fu_variants_SebatLab.tsv"), sep="\t")
    df2 = pd.read_csv(os.path.join(args.tulika_dir, "Fu_variants_LIT.tsv"), sep="\t")
    df = pd.concat([df1, df2], ignore_index=True)

    # Build gene-level variant IDs
    gene_ids = set()
    gene_vt_ids = set()
    for _, row in df.iterrows():
        aas = row["Amino_acids"].split("/")
        vt_id = row["SYMBOL"] + " " + aas[0] + str(int(row["Protein_position"])) + aas[1]
        gene_vt_ids.add(vt_id)
        gene_ids.add(row["SYMBOL"])
    print(f"  {len(gene_vt_ids)} unique gene-level variants across {len(gene_ids)} genes")

    # Gene → UniProt mapping
    if args.tulika_gene_to_uniprot and os.path.exists(args.tulika_gene_to_uniprot):
        with open(args.tulika_gene_to_uniprot, "rb") as f:
            gene_symbol_to_uniprot = pickle.load(f)
        print(f"  Loaded {len(gene_symbol_to_uniprot)} gene→UniProt mappings")
    else:
        print(f"  Fetching HGNC info for {len(gene_ids)} genes…")
        df_hgnc = fetch_hgnc_info(list(gene_ids))
        gene_symbol_to_uniprot = build_gene_to_uniprot(df_hgnc)
        pkl_out = os.path.join(output_dir, "tulika_gene_symbol_to_uniprot.pkl")
        with open(pkl_out, "wb") as f:
            pickle.dump(gene_symbol_to_uniprot, f)

    uniprot_wts = set(gene_symbol_to_uniprot.values())

    # BioGRID intersection
    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)
    all_complexes = get_complexes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq)
    print(f"  BioGRID complexes: {len(all_complexes)}")

    all_uids = {u for pair in all_complexes for u in pair}
    id_to_seq = {uid: uniprot_to_seq[uid] for uid in all_uids if uid in uniprot_to_seq}

    # Insert variant sequences
    for vt_id in gene_vt_ids:
        gene_id, variant = vt_id.split(" ")
        if gene_id not in gene_symbol_to_uniprot:
            continue
        uid = gene_symbol_to_uniprot[gene_id]
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
        new_seq = list(id_to_seq[uid])
        new_seq[mt_idx] = mt_res
        key = f"{uid} {variant}"
        if key in id_to_seq:
            assert id_to_seq[key] == "".join(new_seq)
        else:
            id_to_seq[key] = "".join(new_seq)

    sanity_check(all_complexes, id_to_seq, uniprot_to_interactors)

    complexes_cleaned = clean_complexes(all_complexes)
    for a, b in complexes_cleaned:
        assert not ((b, a) in complexes_cleaned and a != b)

    all_variants = build_variant_triplets(id_to_seq, complexes_cleaned)
    print(f"  Variant triplets: {len(all_variants)}")

    # Save
    with open(f"{output_dir}/all_complexes.pkl", "wb") as f:
        pickle.dump(complexes_cleaned, f)
    with open(f"{output_dir}/id_to_seq.pkl", "wb") as f:
        pickle.dump(id_to_seq, f)
    with open(f"{output_dir}/all_variants.pkl", "wb") as f:
        pickle.dump(all_variants, f)
    print(f"  Outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    mode = args.mode
    if mode in ("neurodev", "both"):
        out = args.neurodev_output_dir or args.output_dir
        run_neurodev(args, out)
    if mode in ("tulika", "both"):
        out = args.tulika_output_dir or args.output_dir
        run_tulika(args, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Map autism/NDD variants to BioGRID PPI network"
    )
    p.add_argument("--mode", choices=["neurodev", "tulika", "both"], default="both",
                   help="Which dataset(s) to process")
    # NeuroDev inputs
    p.add_argument("--neurodev-case", default=None,
                   help="NeuroDev case FASTA (neurodev_case.fasta)")
    p.add_argument("--neurodev-control", default=None,
                   help="NeuroDev control FASTA (neurodev_control.fasta)")
    p.add_argument("--neurodev-refseq-to-uniprot", default=None,
                   help="UniProt mapping TSV for NeuroDev RefSeq IDs")
    p.add_argument("--max-complex-subset", type=int, default=720,
                   help="Max complexes in NeuroDev capped subset (default: 720)")
    # Tulika inputs
    p.add_argument("--tulika-dir", default=None,
                   help="Directory with Fu_variants_SebatLab.tsv and Fu_variants_LIT.tsv")
    p.add_argument("--tulika-fasta", default=None,
                   help="UniProt FASTA for Tulika autism proteins (optional for seq lookup)")
    p.add_argument("--tulika-gene-to-uniprot", default=None,
                   help="Pre-built gene→UniProt pickle for Tulika genes")
    # BioGRID
    p.add_argument("--biogrid-dir", required=True,
                   help="Directory with BioGRID pickles from get_biogrid_interactors.py")
    # Outputs
    p.add_argument("--output-dir", default=None,
                   help="Fallback output directory (used when mode-specific dirs not set)")
    p.add_argument("--neurodev-output-dir", default=None,
                   help="Output directory for NeuroDev results")
    p.add_argument("--tulika-output-dir", default=None,
                   help="Output directory for Tulika/Fu autism results")
    main(p.parse_args())
