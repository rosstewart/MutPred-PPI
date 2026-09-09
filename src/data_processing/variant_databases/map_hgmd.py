#!/usr/bin/env python
"""Map HGMD disease-mutation variants to the BioGRID direct-binding PPI network.

LICENSING NOTE: The HGMD file is a licensed resource. This script reads the
HGMD FASTA as an input file but writes ONLY protein/interaction mapping data
(UniProt IDs, complex pairs, variant counts) to output — no HGMD variant-level
data (amino-acid changes, positions, etc.) is embedded in or emitted by this
script.

The script produces:
  uniprot_wts.pkl         : set of UniProt IDs with HGMD variants in BioGRID
  all_complexes.pkl       : set of (uniprot_a, uniprot_b) complex pairs
  complex_subset.pkl      : capped subset (≤10 partners per protein, ≤600 total)
  id_to_seq.pkl           : WT+VT sequences keyed by UniProt (or 'UniProt variant')
  all_variants.pkl        : set of (uniprot, variant, partner) triplets
  variant_subset.pkl      : triplets for complex_subset

Usage:
    python map_hgmd.py \
        --hgmd-file /path/to/annovar/hgmd_pedja_ethnicity/hgmd_pedja_ethnicity_proteins.fasta \
        --hgmd-dm-wts /path/to/annovar/hgmd_pedja_ethnicity/all_wts_dm.pkl \
        --hgmd-dm-vts /path/to/annovar/hgmd_pedja_ethnicity/all_vts_dm.pkl \
        --refseq-to-uniprot hgmd/hgmd_refseq_to_uniprot.tsv \
        --biogrid-dir biogrid \
        --output-dir $MUTPRED_DATA_ROOT/hgmd \
        --max-complex-subset 600
"""

import argparse
import os
import pickle
from Bio import SeqIO
from biogrid_common import (  # shared verbatim helpers
    build_variant_triplets,
    clean_complexes,
    get_complexes_in_biogrid,
    load_biogrid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_refseq_to_uniprot(tsv_path):
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    mapping = {}
    reviewed = {}
    for _, row in df.iterrows():
        rid, uid, rev = row["From"], row["Entry"], row["Reviewed"]
        is_rev = rev == "reviewed"
        if rid not in mapping or (is_rev and not reviewed.get(rid, False)):
            mapping[rid] = uid
            reviewed[rid] = is_rev
    return mapping


def build_complex_subset(all_complexes, max_size):
    complex_subset = set()
    cleaned = set()
    for a, b in all_complexes:
        count = 0
        if (b, a) not in cleaned:
            cleaned.add((a, b))
        for na, nb in all_complexes:
            if na == a and count < 10:
                if (nb, na) not in complex_subset:
                    complex_subset.add((na, nb))
                    count += 1
                    if len(complex_subset) == max_size:
                        return complex_subset
    return complex_subset


def apply_variants(uniprot_seq_dict, id_to_seq):
    """Write variant sequences into id_to_seq for proteins already present (WT check)."""
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
        new_seq = list(id_to_seq[uid])
        new_seq[mt_idx] = mt_res
        new_seq = "".join(new_seq)
        key = f"{uid} {variant}"
        if key in id_to_seq:
            assert id_to_seq[key] == new_seq
        else:
            id_to_seq[key] = new_seq


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read HGMD FASTA — DM variants only (no variant data retained)
    # ------------------------------------------------------------------
    print(f"Reading HGMD FASTA from {args.hgmd_file} …")
    seq_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.hgmd_file, "fasta")}
    print(f"  {len(seq_dict)} sequences in HGMD FASTA")

    # Load DM-only ID sets
    with open(args.hgmd_dm_wts, "rb") as f:
        all_wts_dm = pickle.load(f)
    with open(args.hgmd_dm_vts, "rb") as f:
        all_vts_dm = pickle.load(f)

    # Filter to DM, converting 'REFSEQ_GENE_VARIANT' keys → 'REFSEQ VARIANT'
    seq_dict_dm = {}
    for seq_id, seq in seq_dict.items():
        parts = seq_id.split("_")
        if len(parts) == 2:
            # WT entry
            if seq_id in all_wts_dm:
                seq_dict_dm[seq_id] = seq
        elif len(parts) == 3:
            # VT entry
            wt = "_".join(parts[:2])
            vt = parts[2]
            if f"{wt} {vt}" in all_vts_dm:
                seq_dict_dm[f"{wt} {vt}"] = seq
    del seq_dict

    all_refseq_ids = {k for k in seq_dict_dm if " " not in k}
    print(f"  DM WT proteins: {len(all_refseq_ids)}")

    # Write RefSeq ID list for manual UniProt mapping
    ids_out = os.path.join(args.output_dir, "all_refseq_ids.txt")
    with open(ids_out, "w") as f:
        f.write(" ".join(all_refseq_ids))
    print(f"  Wrote RefSeq IDs to {ids_out}")
    print("  (If hgmd_refseq_to_uniprot.tsv does not exist yet, map these IDs via UniProt "
          "ID Mapping, then rerun.)")

    # ------------------------------------------------------------------
    # 2. Map RefSeq → UniProt
    # ------------------------------------------------------------------
    refseq_to_uniprot = load_refseq_to_uniprot(args.refseq_to_uniprot)
    print(f"  {len(refseq_to_uniprot)} RefSeq→UniProt mappings loaded")

    uniprot_seq_dict_dm = {}
    for seq_id, seq in seq_dict_dm.items():
        if " " not in seq_id:
            # WT
            if seq_id in refseq_to_uniprot:
                uniprot_seq_dict_dm[refseq_to_uniprot[seq_id]] = seq
        else:
            wt, vt = seq_id.split(" ")
            if wt in refseq_to_uniprot:
                uniprot_seq_dict_dm[f"{refseq_to_uniprot[wt]} {vt}"] = seq

    uniprot_wts = {k for k in uniprot_seq_dict_dm if " " not in k}
    uniprot_vts = {k for k in uniprot_seq_dict_dm if " " in k}
    print(f"  UniProt WTs: {len(uniprot_wts)}, VTs: {len(uniprot_vts)}")

    # Save UniProt WT set (no variant-level data)
    with open(f"{args.output_dir}/uniprot_wts.pkl", "wb") as f:
        pickle.dump(uniprot_wts, f)
    with open(f"{args.output_dir}/hgmd_uniprot_vts.pkl", "wb") as f:
        pickle.dump(uniprot_vts, f)

    # ------------------------------------------------------------------
    # 3. Map to BioGRID and build complex sets
    # ------------------------------------------------------------------
    print("Loading BioGRID network…")
    uniprot_to_interactors, uniprot_to_seq = load_biogrid(args.biogrid_dir)

    all_complexes = get_complexes_in_biogrid(uniprot_wts, uniprot_to_interactors, uniprot_to_seq)
    print(f"BioGRID complexes for HGMD proteins: {len(all_complexes)}")

    all_uids = {u for pair in all_complexes for u in pair}
    id_to_seq = {uid: uniprot_to_seq[uid] for uid in all_uids if uid in uniprot_to_seq}

    # Inject variant sequences
    apply_variants(uniprot_seq_dict_dm, id_to_seq)
    print(f"id_to_seq entries: {len(id_to_seq)}")

    # Sanity check
    for a, b in all_complexes:
        assert (a in uniprot_to_interactors and b in uniprot_to_interactors
                and b in uniprot_to_interactors[a] and a in uniprot_to_interactors[b])

    complexes_cleaned = clean_complexes(all_complexes)
    for a, b in complexes_cleaned:
        assert not ((b, a) in complexes_cleaned and a != b)

    complex_subset = build_complex_subset(all_complexes, args.max_complex_subset)
    print(f"Complexes — cleaned: {len(complexes_cleaned)}, subset: {len(complex_subset)}")

    # ------------------------------------------------------------------
    # 4. Build variant triplets (protein-level mapping only)
    # ------------------------------------------------------------------
    all_variants = build_variant_triplets(id_to_seq, complexes_cleaned)
    variant_subset = build_variant_triplets(id_to_seq, complex_subset)
    print(f"Variant triplets — all: {len(all_variants)}, subset: {len(variant_subset)}")

    # ------------------------------------------------------------------
    # 5. Save outputs — NO variant-level HGMD data written
    # ------------------------------------------------------------------
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

    print(f"\nSummary")
    print(f"  HGMD DM proteins mapped to BioGRID: {len(uniprot_wts & set(u for u,_ in all_complexes))}")
    print(f"  Unique BioGRID complexes: {len(complexes_cleaned)}")
    print(f"  Variant-partner triplets (all): {len(all_variants)}")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Map HGMD DM variants to BioGRID PPI network (no variant data in output)"
    )
    p.add_argument("--hgmd-file", required=True,
                   help="HGMD proteins FASTA (licensed; not embedded in output)")
    p.add_argument("--hgmd-dm-wts", required=True,
                   help="Pickle of DM-only WT RefSeq IDs (all_wts_dm.pkl)")
    p.add_argument("--hgmd-dm-vts", required=True,
                   help="Pickle of DM-only 'RefSeq variant' strings (all_vts_dm.pkl)")
    p.add_argument("--refseq-to-uniprot", required=True,
                   help="UniProt ID mapping TSV for HGMD RefSeq IDs")
    p.add_argument("--biogrid-dir", required=True,
                   help="Directory with BioGRID pickles from get_biogrid_interactors.py")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for HGMD mapping results")
    p.add_argument("--max-complex-subset", type=int, default=600,
                   help="Maximum complexes in the capped subset (default: 600)")
    main(p.parse_args())
